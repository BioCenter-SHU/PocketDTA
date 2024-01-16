import os
import json
import torch
from rdkit import Chem
from tqdm import tqdm
import pandas as pd
import pickle
from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.PDBBind")
@utils.copy_args(data.ProteinLigandDataset.load_sequence)
class PDBBind(data.ProteinLigandDataset):
    """
    The PDBbind-2020 dataset with 19347 binding affinity indicating the interaction strength 
    between pairs of protein and ligand.

    Statistics:
        - #Train: 18022
        - #Valid: 962
        - #Test: 363
    Parameters:
        path (str): path to store the dataset 
        drug_method (str): Drug loading method, from 'smile','2d' or '3d'. 
        protein_method (str): Protein loading method, from 'sequence','pdb'.
        transform (Callable, optional): protein sequence transformation function
        lazy (bool, optional): if lazy mode is used, the protein-ligand pairs are processed in the dataloader.
            This may slow down the data loading process, but save a lot of CPU memory and dataset loading time.
        **kwargs
    """

    splits = ["train", "valid", "test"]
    target_fields = ["affinity"]

    def __init__(self, path='../../data/dta-datasets/PDBbind/', drug_method='smile', protein_method='sequence',
                 transform=None, lazy=False, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.transform = transform
        self.lazy = lazy
        self.protein_method = protein_method  # added for get_item method
        local_file = self.path + 'pdbbind_datasets.csv'
        # self.num_samples = [18028, 963, 363]  # pocket after
        self.num_samples = [18022, 962, 363]  # original
        dataset_df = pd.read_csv(local_file) 

        # ============================== Get the col info of the Dataframe ============================== #
        drug_list = dataset_df["Drug_Index"].tolist() # for further index the drug in 68 durg list
        protein_list = dataset_df["Protein_Index"].tolist()  # for further index the protein in 442 protein list
        self.smiles = dataset_df["Drug"].tolist()
        self.targets = {}  # follow the torchdrug form create a dict to store
        self.targets["affinity"] = dataset_df["Y"].tolist()  # scaled to [0, 1]

        # ============================== Loading ligand position file ==============================  #
        coords_pkl = path + '3d_Molecule.pkl'
        with utils.smart_open(coords_pkl, "rb") as fin:
              coords_list = pickle.load(fin)
        # ============================== Generating the self.data list [protein, mol] ============================== #
        num_sample = len(self.smiles)                
        for field, target_list in self.targets.items():
            if len(target_list) != num_sample:
                raise ValueError("Number of target `%s` doesn't match with number of molecules. "
                                 "Expect %d but found %d" % (field, num_sample, len(target_list)))
        
        self.data = []
        print(f'==================== Using {drug_method} drug method and {protein_method} protein method! ====================')
        # ============================== Loading Protein ============================== #
        protein_pkl = path + protein_method + '_Protein.pkl'
        # ============================== Pkl file not exist, Creating one ==============================  #
        if not os.path.exists(protein_pkl):
            protein_file = self.path + 'pdbbind_proteins.csv'
            # 'gearnet' or 'comenet' are pre-defined through the graph construction
            if protein_method == 'pdb' or 'sequence' or 'gearnet' or 'comenet':
                self.load_protein(protein_file, protein_method, **kwargs)
            else:
                raise ValueError("Protein method should be 'pdb', 'sequence' 'gearnet' or 'comenet'!")
        
        # ============================== Loading Pkl file ==============================  #
        with utils.smart_open(protein_pkl, "rb") as fin:
            target_protein  = pickle.load(fin)
        # ============================== Loading Drug ============================== #
        drug_pkl = path + drug_method + '_Molecule.pkl'
        # ============================== Pkl file not exist, Creating one ==============================  #
        if not os.path.exists(drug_pkl):
            drug_file = self.path + 'pdbbind_ligands.csv'
            if drug_method == 'smile' or '2d' or '3d' or 'comenet':
                self.load_drug(drug_file, drug_method, **kwargs)
            else:
                raise ValueError("Drug method should be 'smile' , '2d' '3d' or 'comenet' !")
        with utils.smart_open(drug_pkl, "rb") as fin:
            drug_molcule  = pickle.load(fin)

        indexes = range(num_sample)
        indexes = tqdm(indexes, "Constructing Dataset from pkl file: ")
        # get the index of protein and mol to quick build
        for i in indexes:
            protein = target_protein[protein_list[i]]
            mol = drug_molcule[drug_list[i]]
            md = coords_list[drug_list[i]]
            self.data.append([protein, mol, md])
        # ============================== Dataset Completing! ============================== #

    # ============================== Surface points for collision detection ============================== #
    def smooth_distance_function(self, atom_coords, point_coords, atom_types, smoothness=0.01):
        """
        @Info   dMaSIF smooth distance function impl
        @Date   2023.12.01
        @Author Marine
        Step 1. Computes a smooth distance from points to all the atoms of a protein.
        Implements Formula 1: SDF(x) = -(B/C) * D, where
        - B = Σₖ₌₁ᴬ exp(-‖x-aₖ‖)*σₖ
        - C = Σₖ₌₁ᴬ exp(-‖x-aₖ‖)
        - D = logΣₖ₌₁ᴬ exp(-‖x-aₖ‖/σₖ)
        Args:
            atom_coords (Tensor): (atom_nums, 3) atom coords of the protein (Red circulars in Figure).
            point_coords (Tensor): (point_nums, 3) sampled points (Blue dots in Figure) of the target surface.
            atom_types (Tensor): (atom_nums, atom_types) one-hot encoding of the 17 different chemical types,
                                 atom_vocab = ["H", "B", "C", "N", "O", "F", "Mg", "Si", "P", "S", "Cl", "Cu", "Zn", "Se", "Br", "Sn", "I"]
        Returns:
            Tensor: (point_nums,) computed smooth distances between sampled points(`y`) and target surface(interenced by `x`).
        """
        # ‖x-aₖ‖ is the distance between all atom a_k and point x
        distance_matrix = ((atom_coords.unsqueeze(dim=1) - point_coords.unsqueeze(dim=0)) ** 2).sum(
            dim=-1).sqrt()  # (atom_nums, point_nums, 1)

        # σₖ is the radii of the current atom with smoothness
        atom_radii = torch.Tensor(
            [1.10, 1.92, 1.70, 1.55, 1.52, 1.47, 1.73, 2.10, 1.80, 1.80,
             1.75, 1.40, 1.39, 1.90, 1.85, 2.17, 1.98],
            device=atom_coords.device).unsqueeze(dim=1)
        atomtype_radii = atom_types @ atom_radii  # (N, 17) @ (17, 1) -> (N, 1)
        atomtype_radii = smoothness * atomtype_radii

        # B = Σₖ₌₁ᴬ exp(-‖x-aₖ‖)*σₖ
        B = ((-distance_matrix).exp() * atomtype_radii).sum(dim=0)

        # C = Σₖ₌₁ᴬ exp(-‖x-aₖ‖)
        C = (-distance_matrix).exp().sum(dim=0)

        # D = logΣₖ₌₁ᴬ exp(-‖x-aₖ‖/σₖ)
        D = (-distance_matrix / atomtype_radii).logsumexp(dim=0)

        smooth_distance = -(B / C) * D

        return smooth_distance

    def subsample(self, point_coords, batch_size=None, scale=[1.0, 1.0, 1.0]):
        """
        Subsamples the point cloud using a grid (cubic) clustering scheme.
        The function returns one average sample per cell, as described in Fig.3e of the paper.
        Args:
            point_coords (Tensor): (point_nums, 3) sampled points (Blue dots in Figure) of the target surface.
            scale (list of float, optional): side length of the cubic grid cells. Defaults to 1 (Angstrom).
        Returns:
            final_coords (Tensor): (final_point_nums, 3): sub-sampled point cloud, with final_point_nums <= point_nums.
        """
        if batch_size is None:  # Single protein case:
            size = torch.tensor(scale, dtype=torch.float32)
            # the index(start from 1) of each grid cube the points belong to
            grid_index = grid_cluster(point_coords, size)
            # get the average point coords in the same grid
            final_coords = scatter_mean(point_coords, grid_index, dim=0)
            return final_coords
        # We process PackedProtein with a for loop.
        else:
            batch_size = torch.max(batch_size).item() + 1  # Typically, =32
            points, batches = [], []
            for b in range(batch_size):
                p = subsample(point_coords[batch_size == b], scale=scale)
                points.append(p)
                batches.append(b * torch.ones_like(batch_size[: len(p)]))
        return torch.cat(points, dim=0), torch.cat(batches, dim=0)

    def atoms_to_points(self, atom_coords, atom_types, target_distance=1.05, sample_ratio=20, iter_nums=5, variance=0.1):
        # Step 2. random generate the points coords
        atom_nums = atom_coords.shape[0]
        point_coords = (
                atom_coords.unsqueeze(dim=1) + 10 * target_distance * torch.randn(atom_nums, sample_ratio, 3).
                type_as(atom_coords).view(-1, 3))
        # avoid the grad too far
        atom_coords = atom_coords.detach().contiguous()
        point_coords = point_coords.detach().contiguous()
        # Step 3. grad the points into the surface
        with torch.enable_grad():
            if point_coords.is_leaf:
                point_coords.requires_grad = True
            for it in range(iter_nums):
                smooth_distance = smooth_distance_function(atom_coords, point_coords, atom_types=atom_types)
                Loss = ((smooth_distance - target_distance) ** 2).sum()
                grad_output = torch.autograd.grad(Loss, point_coords)[0]
                point_coords.data -= 0.5 * grad_output

            # Step 4. clean the points out of margin
            clean_distance = smooth_distance_function(atom_coords, point_coords, atom_types=atom_types)
            margin = (clean_distance - target_distance).abs()
            mask = margin < target_distance * variance
            # Step 4. clean the inner points
            clean_coords = point_coords.detach()
            clean_coords.requires_grad = True
            for it in range(iter_nums):
                smooth_distance = smooth_distance_function(atom_coords, clean_coords, atom_types=atom_types)
                Loss = smooth_distance.sum()
                grad_output = torch.autograd.grad(Loss, clean_coords)[0]
                normals = torch.nn.functional.normalize(grad_output, p=2, dim=-1)
                # 求得法向量，通过沿着最大的方向运动1.5倍距离，若还在内部就说明该店不行
                clean_coords = clean_coords + target_distance * normals
            clean_distance = smooth_distance_function(atom_coords, clean_coords, atom_types=atom_types)
            mask = mask & (clean_distance > 1.5 * target_distance)
            point_coords = point_coords[mask].contiguous().detach()
            # Step 5. subsample the uniform density
            final_coords = subsample(point_coords)
            # Step 6. compute the normals of the final points
            normal_points = final_coords.detach()
            normal_points.requires_grad = True
            smooth_distance = smooth_distance_function(atom_coords, normal_points, atom_types=atom_types)
            Loss = (1.0 * smooth_distance).sum()
            grad_output = torch.autograd.grad(Loss, normal_points)[0]
            normals = torch.nn.functional.normalize(grad_output, p=2, dim=-1)  # get the normal vector
            final_coords = final_coords - 0.5 * normals
            return final_coords.detach(), normals.detach()

    # fixed index as the DeepDTA provided, we need to reproduce the 5 folds for training, 1 fold for test
    def deepdta_split(self):
        train_index = json.load(open(f"{self.path}train_index.txt"))
        valid_index = json.load(open(f"{self.path}valid_index.txt"))
        test_index = json.load(open(f"{self.path}test_index.txt"))  
        train_set = torch.utils.data.Subset(self, train_index) 
        valid_set = torch.utils.data.Subset(self, valid_index)
        test_set = torch.utils.data.Subset(self, test_index)
        return train_set, valid_set, test_set

    # random split 4:1:1 as the num_samples
    def random_split(self):  
        train_set, valid_set, test_set = torch.utils.data.random_split(self, self.num_samples)
        return train_set, valid_set, test_set 

    def get_item(self, index):
        if self.lazy:
            graph1 = data.Protein.from_pdb(self.pdb_files[index], self.kwargs)
            mol = Chem.MolFromSmiles(self.smiles[index])
            if not mol:
                graph2 = None
            else:
                graph2 = data.Molecule.from_molecule(mol, **self.kwargs)
        else:
            graph1 = self.data[index][0]
            graph2 = self.data[index][1]
            graph3 = self.data[index][2]
        if hasattr(graph1, "residue_feature"):
            with graph1.residue():
                graph1.residue_feature = graph1.residue_feature.to_dense()
        # # 将坐标的标签加入属性，就不会在运行时报错，因为batch的堆叠要求tensor维度得一致
        # with graph2.atom():
        #     graph2.coords = self.drug_coords[index] # trans the dim for batch pack

        item = {"graph1": graph1,
                "graph2": graph2,
                "graph3": graph3}
        item.update({k: v[index] for k, v in self.targets.items()})
        if self.transform:
            item = self.transform(item)
        return item