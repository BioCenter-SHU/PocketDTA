# 基于蛋白质口袋的药物-靶标亲和力预测模型研究

## 代码说明

> 本论文的代码由第三章PocketDTA和第四章EMPDTA组成，值得说明的是，两个模型均是在TorchDrug上进行的二次开发，细节会在后面详细介绍

其中[TorchDrug](https://torchdrug.ai/)的网址上有着对于原生代码运行的tutorial，也可以参考本地提供的`DTA_WORK/script/notebook/tutorial/torchprotein_tutorial1.ipynb`进行学习和梳理

### TorchDrug API

> https://torchdrug.ai/docs/notes/graph.html

As all the drugs and proteins in torchdrug are condsidered as graphs. The core is to add feature into the graph attribute. So the first thing is to get clear how the data flow in the PLI task.

> Model Reference https://torchdrug.ai/docs/notes/model.html
> Dataset Reference https://github.com/DeepGraphLearning/torchdrug/issues/27

The first thing is to confirm the dataset that all researchers used. 
All I know is that Davis, KIBA and BindingDB are the common dataset.
As far as know, the torchdrug only containing part of BindingDB dataset.
It is necessary to build my own dataset containing the same data baseline has.

### 代码书写格式

My code will reference the Peer Benchmark(2022) format which also is the torchdrug publication.

> Reference: PEER Benchmark
> - Python file for DTA Structure: Multi-Prediction.py
> - Config file for different combination: CNN + GIN.yaml

In more details, the main script will be only one python file. The different model run experiments will be recorded in config yaml. In that way, code will be consistant and simple.

## 数据说明

### PKL File 文件格式说明

The GearNet use the protein graph in pkl file format. The python.pickle packge is the serialization tool. Because the data in memory will lose when the power is off, it is necessary to store the data as files in the disks. And pickle is python only, object and function can be stored both. So it's useful to temporary store the whole python output as pkl file.
因此，本文的实验均是以`xxx.pkl`的形式将Python对象进行存储，方便在构建数据集的时候直接读取，当然从公开的数据集到供模型使用的包含TorchDrug对象的pkl文件的构建过程也会在`DTA_WORK/script/notebook/DatasetBuilding`中进行详细介绍。

## 代码运行

### 数据预处理

从公开的原始数据集（Filtered Davis，Davis，KIBA和PDBbind）出发，通过对应的模态补全和图结构构建notebook代码就能得到TorchDrug的Protein和Molecule对象，为了加速训练本文将药物和蛋白质的数据按照列表的方式以pkl文件形式存储。

Scripts for all four dataset construction are provided in DatasetBuilding in the script/notebook/DatasetBuilding/ file. Davis dataset for example, three main notebook file need for generating DTA, protein, drug pkl file.

- DavisDataset.ipynb 用于Davis数据集从原始数据构建，生成CSV文件
- DavisESM.ipynb 通过CPU将蛋白质的残基输入到ESM-2b模型并输出对应残基的1028维特征
- DavisMolecule.ipynb 通过MolFormer预训练模型将对应SMILES

The file torchdrug/datasets/davis.py in torchdrug then provide the whole picture for Davis and Filtered Davis dataset.

```python
import os
import json
import random
from tqdm import tqdm
import pickle
import pandas as pd
from rdkit import Chem
from copy import deepcopy
from collections import defaultdict
import torch
from torch.utils import data as torch_data
from torchdrug import data, utils, core
from torchdrug.core import Registry as R

# Davis数据集的TorchDrug处理流程，通过参数读取PKL文件并根据索引得到一对药物-靶标亲和力数据（protein和mol）并预测Y
@R.register("datasets.Davis")
@utils.copy_args(data.ProteinLigandDataset.load_sequence)
class Davis(data.ProteinLigandDataset):
    """ 
    Input: 
        path = "../../data/dta-datasets/Davis/"
        method = 'sequence', 'pdb' or 'sdf' for different init instance methods.
    davis_datasets.csv ==> pd.series containing the all the col for usage!
    ["Drug"], ["Target"], ["Y"] and ["PDB_File"] ==> .tolist() for the input form list [].
    load_sequence/load_pdb/load_sdf function to build the dataset containing different attritubes.
    Notes: the class attributes do not need to be defined first. Once used they will be defined.
    - pdb_files: for protein 3d pdb file name and location
    - sequences: for protein 1d sequence
    - sdf: for drug 3d position
    - smiles: for drug 1d smiles
    - targets: for the affinity scores
    - data: for the torchdrug tuple(Protein, Molecule) for DTA task
    The interaction of 68(filtered protein under 10 interactions) kinase inhibitors with 442 kinases 
    covering>80% of the human catalytic protein kinome.(only 379 kinases are unique).
    Statistics for the 'whole' Dataset:
        - #Molecule: 68
        - #Protein: 442
        - #Interaction: 30,056
        - split:
        train:valid:test = 4:1:1 (and test shoule be hold out)
        - #Train: 20036
        - #Valid: 5010
        - #Test: 5010
    Parameters:
        path (str): path to store the dataset 
        drug_method (str): Drug loading method, from 'smile','2d' or '3d'. 
        protein_method (str): Protein loading method, from 'sequence','pdb'.
        description (str): whole(30056) or filter(9125) davis dataset
        transform (Callable, optional): protein sequence transformation function
        lazy (bool, optional): if lazy mode is used, the protein-ligand pairs are processed in the dataloader.
            This may slow down the data loading process, but save a lot of CPU memory and dataset loading time.
        **kwargs
    """
    splits = ["train", "valid", "test"]
    target_fields = ["affinity"]
    # Three init way, first the normal way using sequences,
    # another using the pdb/sdf for protein smiles for drug
    def __init__(self, path='../../data/dta-datasets/Davis/', drug_method='smile', protein_method='sequence',
                description='whole', transform=None, lazy=False, **kwargs):
        """
        - Use the DeepDTA davis dataset to create the local csv file.
        - Then init the Davis dataset instance by loading the local file to pd.DataFrame.
        - Select the col of the df to build the dataset object.
        """   
        # ============================== Path check and generate ============================== # 
        # os.path.expanduser expand the ~ to the full path
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.lazy = lazy
        self.transform = transform
        self.protein_method = protein_method  # added for get_item method
        self.description = description
        # ============================== Choose the local file for part/whole dataset ============================== #
        if self.description == 'filter':
            # 9,125 samples
            local_file = self.path + 'davis_filtered_datasets.csv'
            self.num_samples = [6085, 1520, 1520] 
        else:
            # 30,056 samples
            local_file = self.path + 'davis_datasets.csv'
            self.num_samples = [20036, 5010, 5010] 
        dataset_df = pd.read_csv(local_file)      
             
        # ============================== Get the col info of the Dataframe ============================== #
        drug_list = dataset_df["Drug_Index"].tolist() # for further index the drug in 68 durg list
        protein_list = dataset_df["protein_Index"].tolist()  # for further index the protein in 442 protein list
        self.pdb_files = dataset_df["PDB_File"].tolist()  # pdb file name like AF-Q2M2I8-F1-model_v4.pdb
        self.smiles = dataset_df["Drug"].tolist()
        self.targets = {}  # follow the torchdrug form create a dict to store
        self.targets["affinity"] = dataset_df["Y"].tolist()  # scaled to [0, 1]

        # ============================== Loading label file ==============================  #
        label_pkl = path + 'gearnet_labeled.pkl'
        with utils.smart_open(label_pkl, "rb") as fin:
            label_list = pickle.load(fin)

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
            protein_file = self.path + 'davis_proteins.csv'
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
        if not os.path.exists(drug_pkl):
            drug_file = self.path + 'davis_ligands.csv'
            if drug_method == 'smile' or '2d' or '3d' or 'comenet':
                self.load_drug(drug_file, drug_method, **kwargs)
            else:
                raise ValueError("Drug method should be 'smile' , '2d' '3d' or 'comenet' !")
        with utils.smart_open(drug_pkl, "rb") as fin:
            drug_molcule  = pickle.load(fin)

        # map the 442 protein label into 9125 DTA pairs
        self.label_list = []
        indexes = range(num_sample)
        indexes = tqdm(indexes, "Constructing Dataset from pkl file: ")
        # get the index of protein and mol to quick build
        for i in indexes:
            protein = target_protein[protein_list[i]]
            mol = drug_molcule[drug_list[i]]
            self.label_list.append(label_list[protein_list[i]])
            self.data.append([protein, mol])
        # ============================== Dataset Completing! ============================== #

    # 数据集的划分方式，这是实现了多种不同的数据集划分方式，可以在代码中选择
    # sequence split
    def split(self, keys=None):  
        keys = keys or self.splits
        offset = 0
        splits = []
        for split_name, num_sample in zip(self.splits, self.num_samples):
            if split_name in keys:
                split = torch_data.Subset(self, range(offset, offset + num_sample))
                splits.append(split)
            offset += num_sample
        return splits

    # random split 4:1:1 as the num_samples
    def random_split(self):  
        train_set, valid_set, test_set = torch.utils.data.random_split(self, self.num_samples)
        return train_set, valid_set, test_set 

    # fixed index as the DeepDTA provided, we need to reproduce the 5 folds for training, 1 fold for test
    def deepdta_split(self, fold):
        if self.description == 'filter':
            train_valid_index = json.load(open("../../data/dta-datasets/Davis/train_fold_setting_for_filter_davis.txt"))
            test_index = json.load(open("../../data/dta-datasets/Davis/test_fold_setting_for_filter_davis.txt"))   
        else:        
            train_valid_index = json.load(open("../../data/dta-datasets/Davis/train_fold_setting1.txt"))
            test_index = json.load(open("../../data/dta-datasets/Davis/test_fold_setting1.txt"))
        test_set = torch.utils.data.Subset(self, test_index)
        print(f'==================== Training on Fold: {fold} ====================')
        train_index = []
        valid_index = []
        valid_index = train_valid_index[fold]  # which is the valid list for dataset build
        otherfolds = deepcopy(train_valid_index)  # copy a new list without effecting the raw list
        otherfolds.pop(fold)  # pop out the valid fold and left the other 4 folds for training
        for train_fold in otherfolds:  # to merge the 4 fold into a single train_index list
            train_index.extend(train_fold)
        # Get the list, now build the dataset
        valid_set = torch.utils.data.Subset(self, valid_index) 
        train_set = torch.utils.data.Subset(self, train_index)
        return train_set, valid_set, test_set

    # Clustered cross-validation 3-Fold
    # from 'Latent Biases in Machine Learning Models for Predicting Binding Affinities Using Popular Data Sets'
    def ccv_split(self, fold, mode="target"):
        if self.description == 'filter' and mode == "target":
            fold0_file = "../../data/dta-datasets/Davis/Davis_CCV_Split/FilteredDavis_target_ccv0.csv"
            fold1_file = "../../data/dta-datasets/Davis/Davis_CCV_Split/FilteredDavis_target_ccv1.csv"
            fold2_file = "../../data/dta-datasets/Davis/Davis_CCV_Split/FilteredDavis_target_ccv2.csv"
        elif self.description == 'whole' and mode == "target":
            fold0_file = "../../data/dta-datasets/Davis/Davis_CCV_Split/Davis_target_ccv0.csv"
            fold1_file = "../../data/dta-datasets/Davis/Davis_CCV_Split/Davis_target_ccv1.csv"
            fold2_file = "../../data/dta-datasets/Davis/Davis_CCV_Split/Davis_target_ccv2.csv"
        fold0_index = pd.read_csv(fold0_file, header=None)
        fold0_index = fold0_index[0].to_numpy().tolist()
        fold1_index = pd.read_csv(fold1_file, header=None)
        fold1_index = fold1_index[0].to_numpy().tolist()
        fold2_index = pd.read_csv(fold2_file, header=None)
        fold2_index = fold2_index[0].to_numpy().tolist()
        if fold == 0:
            train_index = []
            train_index.extend(fold1_index)
            train_index.extend(fold2_index)
            train_set = torch.utils.data.Subset(self, train_index)
            test_set = torch.utils.data.Subset(self, fold0_index)
        elif fold == 1:
            train_index = []
            train_index.extend(fold0_index)
            train_index.extend(fold2_index)
            train_set = torch.utils.data.Subset(self, train_index)
            test_set = torch.utils.data.Subset(self, fold1_index)
        else:
            train_index = []
            train_index.extend(fold0_index)
            train_index.extend(fold1_index)
            train_set = torch.utils.data.Subset(self, train_index)
            test_set = torch.utils.data.Subset(self, fold2_index)
        valid_set = []  # empty
        print(f'==================== Training on Mode {mode} Fold-{fold} ====================')
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
        if hasattr(graph1, "residue_feature"):
            with graph1.residue():
                graph1.residue_feature = graph1.residue_feature.to_dense()
                target = torch.as_tensor(self.label_list[index]["target"], dtype=torch.long)
                graph1.target = target
                mask = torch.as_tensor(self.label_list[index]["mask"], dtype=torch.bool)
                graph1.mask = mask

        item = ({
            "graph1": graph1,
            "graph2": graph2,
        })

        item.update({k: v[index] for k, v in self.targets.items()})
        if self.transform:
            item = self.transform(item)
        return item


@R.register("datasets.FilteredDavis")
@utils.copy_args(data.ProteinLigandDataset.load_sequence)
class FilteredDavis(data.ProteinLigandDataset):
    """
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

    def __init__(self, path='../../data/dta-datasets/Davis/', drug_method='smile', protein_method='sequence',
                 transform=None, lazy=False, **kwargs):
        # ============================== Path check and generate ============================== # 
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.lazy = lazy
        self.transform = transform
        self.protein_method = protein_method  # added for get_item method
        # ============================== Choose the local file for part/whole dataset ============================== #
        # 9,125 samples
        local_file = self.path + 'davis_filtered_datasets.csv'
        self.num_samples = [6085, 1520, 1520] 
        dataset_df = pd.read_csv(local_file)      
             
        # ============================== Get the col info of the Dataframe ============================== #
        drug_list = dataset_df["Drug_Index"].tolist() # for further index the drug in 68 durg list
        protein_list = dataset_df["protein_Index"].tolist()  # for further index the protein in 442 protein list
        self.pdb_files = dataset_df["PDB_File"].tolist()  # pdb file name like AF-Q2M2I8-F1-model_v4.pdb
        self.smiles = dataset_df["Drug"].tolist()
        self.targets = {}  # follow the torchdrug form create a dict to store defaultdict(list)
        self.targets["affinity"] = dataset_df["Y"].tolist()  # scaled to [0, 1]

        # ============================== Loading drug 3d file ==============================  #
        # coords_pkl = path + '3d_Molecule.pkl' # for distance
        # with utils.smart_open(coords_pkl, "rb") as fin:
        #       coords_list = pickle.load(fin)

        # ============================== Loading label file ==============================  #
        label_pkl = path + 'gearnet_labeled.pkl'
        with utils.smart_open(label_pkl, "rb") as fin:
            label_list = pickle.load(fin)

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
            protein_file = self.path + 'davis_proteins.csv'
            # 'gearnet' or 'comenet' are pre-defined through the graph construction
            if protein_method == 'pdb' or 'sequence' or 'gearnet' or 'comenet':
                self.load_protein(protein_file, protein_method, **kwargs)
            else:
                raise ValueError("Protein method should be 'pdb', 'sequence' 'gearnet' or 'comenet'!")
        # ============================== Loading Pkl file ==============================  #
        with utils.smart_open(protein_pkl, "rb") as fin:
            target_protein = pickle.load(fin)

        # ============================== Loading Drug ============================== #
        drug_pkl = path + drug_method + '_Molecule.pkl'
        # ============================== Pkl file not exist, Creating one ==============================  #
        if not os.path.exists(drug_pkl):
            drug_file = self.path + 'davis_ligands.csv'
            if drug_method == 'smile' or '2d' or '3d' or 'comenet':
                self.load_drug(drug_file, drug_method, **kwargs)
            else:
                raise ValueError("Drug method should be 'smile' , '2d' '3d' or 'comenet' !")
        # ============================== Loading Pkl file ==============================  #
        with utils.smart_open(drug_pkl, "rb") as fin:
            drug_molcule  = pickle.load(fin)

        # map the 442 protein label into 9125 DTA pairs
        self.label_list = []
        indexes = range(num_sample)
        indexes = tqdm(indexes, "Constructing Dataset from pkl file: ")
        # get the index of protein and mol to quick build
        for i in indexes:
            protein = target_protein[protein_list[i]]
            drug = drug_molcule[drug_list[i]]
            self.label_list.append(label_list[protein_list[i]])
            self.data.append([protein, drug])

        # ============================== Dataset Completing! ============================== #

    # sequence split
    def split(self, keys=None):  
        keys = keys or self.splits
        offset = 0
        splits = []
        for split_name, num_sample in zip(self.splits, self.num_samples):
            if split_name in keys:
                split = torch_data.Subset(self, range(offset, offset + num_sample))
                splits.append(split)
            offset += num_sample
        return splits

    # random split 4:1:1 as the num_samples
    def random_split(self):  
        train_set, valid_set, test_set = torch.utils.data.random_split(self, self.num_samples)
        return train_set, valid_set, test_set 

    def deepdta_split(self, fold):
        # fixed index as the DeepDTA provided, we need to reproduce the 5 folds for training, 1 fold for test
        train_valid_index = json.load(open("../../data/dta-datasets/Davis/train_fold_setting_for_filter_davis.txt"))
        test_index = json.load(open("../../data/dta-datasets/Davis/test_fold_setting_for_filter_davis.txt"))
        test_set = torch.utils.data.Subset(self, test_index)
        print(f'==================== Training on Fold: {fold} ====================')
        train_index = []
        valid_index = []
        valid_index = train_valid_index[fold]  # which is the valid list for dataset build
        otherfolds = deepcopy(train_valid_index)  # copy a new list without effecting the raw list
        otherfolds.pop(fold)  # pop out the valid fold and left the other 4 folds for training
        for train_fold in otherfolds:  # to merge the 4 fold into a single train_index list
            train_index.extend(train_fold)
        # Get the list, now build the dataset
        valid_set = torch.utils.data.Subset(self, valid_index) 
        train_set = torch.utils.data.Subset(self, train_index)
        return train_set, valid_set, test_set

    # Clustered cross-validation 3-Fold
    # from 'Latent Biases in Machine Learning Models for Predicting Binding Affinities Using Popular Data Sets'
    def ccv_split(self, fold):
        fold0_file = "../../data/dta-datasets/Davis/Davis_CCV_Split/FilteredDavis_target_ccv0.csv"
        fold1_file = "../../data/dta-datasets/Davis/Davis_CCV_Split/FilteredDavis_target_ccv1.csv"
        fold2_file = "../../data/dta-datasets/Davis/Davis_CCV_Split/FilteredDavis_target_ccv2.csv"
        fold0_index = pd.read_csv(fold0_file, header=None)
        fold0_index = fold0_index[0].to_numpy().tolist()
        fold1_index = pd.read_csv(fold1_file, header=None)
        fold1_index = fold1_index[0].to_numpy().tolist()
        fold2_index = pd.read_csv(fold2_file, header=None)
        fold2_index = fold2_index[0].to_numpy().tolist()
        if fold == 0:
            train_index = []
            train_index.extend(fold1_index)
            train_index.extend(fold2_index)
            train_set = torch.utils.data.Subset(self, train_index)
            test_set = torch.utils.data.Subset(self, fold0_index)
        elif fold == 1:
            train_index = []
            train_index.extend(fold0_index)
            train_index.extend(fold2_index)
            train_set = torch.utils.data.Subset(self, train_index)
            test_set = torch.utils.data.Subset(self, fold1_index)
        else:
            train_index = []
            train_index.extend(fold0_index)
            train_index.extend(fold1_index)
            train_set = torch.utils.data.Subset(self, train_index)
            test_set = torch.utils.data.Subset(self, fold2_index)
        valid_set = []  # empty
        print(f'==================== Training on CCV Fold-{fold} ====================')
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
        if hasattr(graph1, "residue_feature"):
            with graph1.residue():
                graph1.residue_feature = graph1.residue_feature.to_dense()
                target = torch.as_tensor(self.label_list[index]["target"], dtype=torch.long)
                graph1.target = target
                mask = torch.as_tensor(self.label_list[index]["mask"], dtype=torch.bool)
                graph1.mask = mask

        item = ({
            "graph1": graph1,
            "graph2": graph2,
        })

        item.update({k: v[index] for k, v in self.targets.items()})
        if self.transform:
            item = self.transform(item)
        return item

# 用于蛋白质口袋的预测内容
@R.register("datasets.DavisProtein")
@utils.copy_args(data.ProteinDataset.load_sequence)
class DavisProtein(data.ProteinDataset):
    """
    The 379 proteins in Davis dataset for pocket label prediction. CCV split are used.
    Parameters:
        path (str): path to store the dataset
        protein_method (str): Protein loading method, from 'sequence','pdb'.
        transform (Callable, optional): protein sequence transformation function.
        **kwargs
    """
    def __init__(self, path='../../data/dta-datasets/Davis/', protein_method='sequence', transform=None, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.transform = transform
        self.data = []

        self.protein_file = self.path + 'davis_proteins.csv'

        # ============================== Pkl file not exist, Creating one ==============================  #
        protein_pkl = path + protein_method + '_Protein.pkl'
        if not os.path.exists(protein_pkl):
            if protein_method == 'pdb' or 'sequence' or 'gearnet' or 'comenet':
                self.load_protein(self.protein_file, protein_method, **kwargs)
            else:
                raise ValueError("Protein method should be 'pdb', 'sequence' 'gearnet' or 'comenet'!")

        # ============================== Loading Pkl file ==============================  #
        with utils.smart_open(protein_pkl, "rb") as fin:
            target_protein = pickle.load(fin)

        # ============================== Loading label file ==============================  #
        label_pkl = path + 'gearnet_labeled.pkl'  # gearnet_labeled pocket_uniprot_label
        with utils.smart_open(label_pkl, "rb") as fin:
            self.label_list = pickle.load(fin)

        # get the index of protein and mol to quick build
        for i in range(len(target_protein)):
            protein = target_protein[i]
            self.data.append(protein)
        # ============================== Dataset Completing! ============================== #

    def random_split(self):
        train_set, valid_set, test_set = torch.utils.data.random_split(self, [147, 147, 148])
        return train_set, valid_set, test_set

    # Clustered cross-validation 3-Fold
    # from 'Latent Biases in Machine Learning Models for Predicting Binding Affinities Using Popular Data Sets'
    def ccv_split(self, fold=0):
        json_file = self.path + "/Davis_CCV_Split/3_folds_target.json"
        fold_dict = json.load(open(json_file))
        protein_df = pd.read_csv(self.protein_file)

        fold0_index = []
        fold1_index = []
        fold2_index = []
        # mapping the name into index of CCV three folds
        for index in range(len(protein_df['Gene'])):
            gene_name = protein_df['Gene'][index]
            if gene_name in fold_dict['fold0']:
                fold0_index.append(index)
            elif gene_name in fold_dict['fold1']:
                fold1_index.append(index)
            elif gene_name in fold_dict['fold2']:
                fold2_index.append(index)

        if fold == 0:
            train_set = torch.utils.data.Subset(self, fold1_index)
            valid_set = torch.utils.data.Subset(self, fold2_index)
            test_set = torch.utils.data.Subset(self, fold0_index)
        elif fold == 1:
            train_set = torch.utils.data.Subset(self, fold0_index)
            valid_set = torch.utils.data.Subset(self, fold2_index)
            test_set = torch.utils.data.Subset(self, fold1_index)
        else:
            train_set = torch.utils.data.Subset(self, fold1_index)
            valid_set = torch.utils.data.Subset(self, fold0_index)
            test_set = torch.utils.data.Subset(self, fold2_index)
        print(f'==================== Training on Fold-{fold} ====================')
        return train_set, valid_set, test_set

    def get_item(self, index):
        graph1 = self.data[index]
        if hasattr(graph1, "residue_feature"):
            with graph1.residue():
                graph1.residue_feature = graph1.residue_feature.to_dense()
                target = torch.as_tensor(self.label_list[index]["target"], dtype=torch.long)
                graph1.target = target
                mask = torch.as_tensor(self.label_list[index]["mask"], dtype=torch.bool)
                graph1.mask = mask
        item = {"graph1": graph1}

        if self.transform:
            item = self.transform(item)
        return item

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))
```

### 运行

The single sun and wandb sweep file can be found in script/pythonfile/, after enter your conda env just use python `SingleRun_davis.py` could run the single run on Davis dataset. The sweep need wandb package for automatic hyperparamers searching.

这里以`SingleRun_davis.py`为例，对代码的运行进行说明，仅需在pythonfile文件下使用命令行激活torchdrug，然后`python SingleRun_davis.py`即可运行。

```python
# 导入依赖库
import os
import random
import torch
import warnings
import wandb
import numpy as np
from torchdrug import datasets, transforms, models, tasks, core

# 固定随机种子，使得结果可复现
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 初始化Conv和Linear的分布，同样使得结果可复现
def initialize_weights(m):
    # Conv1d init
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    # Linear init
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    # BatchNorm1d init
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


seed_torch()
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"  # 更大范围利用显存
warnings.filterwarnings("ignore")

# ============================== dataset loading ============================== #
protein_view = transforms.ProteinView(view="residue", keys="graph1") # 将蛋白质使用残基尺度构造图结构
dataset = datasets.Davis(protein_method="gearnetesm_pocket", drug_method="distanceMol", transform=protein_view) # gearnetesm_pocket是对应的蛋白质pkl文件，distanceMol是对应的药物pkl文件，通过上面介绍的class Davis加载TorchDrug的dataset

train_set, valid_set, test_set = dataset.deepdta_split(fold=0) # 指定deepdta_split方式划分数据集，并指定0折交叉
print(f"Train samples: {len(train_set)}, Valid samples: {len(valid_set)}, Test samples: {len(test_set)}") # 打印数据集信息

# ============================== model define as the pretraining model ============================== #
# 指定蛋白质和药物的结构编码器为GearNet和RGCN，并指定具体的参数
protein_model = models.GearNet(
    input_dim=1280, hidden_dims=[512, 512, 512, 512], num_relation=7, batch_norm=True,
    concat_hidden=True, readout="mean"
)

drug_model = models.RGCN(
    input_dim=67, hidden_dims=[256, 256, 256, 256, 256], num_relation=4, edge_input_dim=19,
    concat_hidden=True, batch_norm=True, readout="mean"
)

# ============================== task define and other training prepare ============================== #
# 指定药物-靶标亲和力预测任务为InteractionPrediction
task = tasks.InteractionPrediction(
    model=protein_model, model2=drug_model, mode_type='MolFormer', task=dataset.tasks, criterion="mse",
    metric=("mse", "c_index", "rm2"), normalization=True, num_mlp_layer=3
)

task.apply(initialize_weights) # 应用上面定义的参数初始化

# ============================== Normal Lr setting ============================== #
# 训练相关记录使用wandb，这里指定项目名称方便归档，初次使用可能需要账号密码
wandb.init(project="Davis_Experiments")  # Davis for final
learning_rate = 2e-4
weight_decay = 2e-4
optimizer = torch.optim.AdamW(params=task.parameters(), lr=learning_rate, weight_decay=weight_decay)
# 将task包一层为solver
solver = core.Engine(
    task, train_set, valid_set, test_set, optimizer, None, gpus=[0], batch_size=128, logger="wandb"
)
# 定义参数优化
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, mode='min', factor=0.2, patience=50, min_lr=1e-5
)

whole_params = sum(p.numel() for p in solver.model.parameters())
print(f'#The Whole Params: {whole_params}')

# ============================== Training Begin ============================== #
# 使用早停策略训练
early_stopping = core.EarlyStopping(patience=100)
checkpoint = "../../result/model_pth/gearnet_d_0513.pth" # 命名本地的pth文件名用于临时保存

# valid performance each epoch
for epoch in range(1000):
    print(">>>>>>>>   Model' LR: ", optimizer.param_groups[0]['lr'])
    solver.train() # 训练单轮
    metric = solver.evaluate("valid")['mean squared error [affinity]'] # 根据mse进行参数优化
    scheduler.step(metrics=metric)  
    # add early stopping
    early_stopping(val_loss=metric, solver=solver, path=checkpoint)
    if early_stopping.early_stop:
        print(">>>>>>>>   Early stopping   >>>>>>>>")
        break
# after all epoches test the performance
solver.load(checkpoint) # 加载最佳模型
solver.evaluate("test") # 在测试集上训练

```