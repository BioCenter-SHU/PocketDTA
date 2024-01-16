import os
import glob

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.AlphaFoldDB")
@utils.copy_args(data.ProteinDataset.load_pdbs)
class AlphaFoldDB(data.ProteinDataset):
    """
    3D protein structures predicted by AlphaFold.
    This dataset covers proteomes of 48 organisms, as well as the majority of Swiss-Prot.

    Statistics:
        See https://alphafold.ebi.ac.uk/download

    Parameters:
        path (str): path to store the dataset
        species_id (int, optional): the id of species to be loaded. The species are numbered
            by the order appeared on https://alphafold.ebi.ac.uk/download (0-20 for model 
            organism proteomes, 21 for Swiss-Prot)
        split_id (int, optional): the id of split to be loaded. To avoid large memory consumption 
            for one dataset, we have cut each species into several splits, each of which contains 
            at most 22000 proteins.
        verbose (int, optional): output verbose level
        **kwargs
    """

    urls = [
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000006548_3702_ARATH_v3.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000001940_6239_CAEEL_v3.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000559_237561_CANAL_v3.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000437_7955_DANRE_v3.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000002195_44689_DICDI_v3.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000803_7227_DROME_v3.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000625_83333_ECOLI_v3.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000008827_3847_SOYBN_v3.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v3.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000805_243232_METJA_v3.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000589_10090_MOUSE_v3.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000059680_39947_ORYSJ_v3.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000002494_10116_RAT_v3.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000002311_559292_YEAST_v3.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000002485_284812_SCHPO_v3.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000007305_4577_MAIZE_v3.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v3.tar"
    ]
    md5s = [
        'd118deb5353ec4beaed6ed9900bf250c', 'f7c8acbda60412395bd7b35bf48428b2',
        '61d31fdb0f46f337c27b155277f6ca30', '0e10ef824b0edde03684ebef767916a7',
        'e8c8550c897d6ca04b2a67cad77a9ab5', '40ca61a10c2120d7ac85ed443f2a94e0',
        '064055cd90509564b22aa670cebd63c2', 'b5dd2486aae6047e82ae33b7af4c8515',
        'ccdfa4629f3a2b734af244900d4bacdc', 'e641031014b85124122fa854010a1fa3',
        '09828386ae2a13d79c02c21b5a0a4ad6', '23716b799aa66c8b073361131a2c75a9',
        '531cb2a204f10902aad6655da5513913', '9f56554d7128f4501ee0ec363d17e528',
        'e589f9bfde1afb2fbeb2d919c075a865', 'b05bd4e542c6f96cc6a9c986750af838',
        '48561a4aed69a93398d2e8bc0d315305'
    ]
    species_nsplit = [
        2, 1, 1, 2, 1, 1, 1, 3, 2, 1,
        1, 2, 1, 1, 1, 3, 20
    ]
    split_length = 22000

    def __init__(self, path, species_id=0, split_id=0, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        species_name = os.path.basename(self.urls[species_id])[:-4]
        if split_id >= self.species_nsplit[species_id]:
            raise ValueError("Split id %d should be less than %d in species %s" % 
                            (split_id, self.species_nsplit[species_id], species_name))
        self.processed_file = "%s_%d.pkl.gz" % (species_name, split_id)
        pkl_file = os.path.join(path, self.processed_file)

        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            print("!!!!!!!!!!!!!!!!!!!! The pkl file: ", self.processed_file, " not exists!!!!!!!!!!!!!!!!!!!!")
            tar_file = utils.download(self.urls[species_id], path, md5=self.md5s[species_id])
            pdb_path = utils.extract(tar_file)
            gz_files = sorted(glob.glob(os.path.join(pdb_path, "*.pdb.gz")))
            pdb_files = []
            index = slice(split_id * self.split_length, (split_id + 1) * self.split_length)
            for gz_file in gz_files[index]:
                pdb_files.append(utils.extract(gz_file))
            self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
            self.save_pickle(pkl_file, verbose=verbose)

        #     tar_file = utils.download(self.urls[species_id], self.path, md5=self.md5s[species_id])
        #     if split_id == 0:
        #         pdb_path = utils.extract(tar_file)

        #     gz_files = sorted(glob.glob(os.path.join(pdb_path, "*.pdb.gz")))
        #     index = slice(split_id * split_length, (split_id + 1) * split_length)
        #     for gz_file in gz_files[index]:
        #         pdb_files.append(utils.extract(gz_file))
        #     protein = ProteinDataset()
        #     protein.load_pdbs(pdb_files=pdb_files, verbose=1)
        #     protein.save_pickle(pkl_file=pkl_file, verbose=1)
        #     print("====================Generated the pkl file", processed_file, "====================")
        # files = os.listdir(path)
        # for file in files:
        #     if file.endswith(('_v3.pdb.gz', '_v3.cif.gz', '_v3.pdb')):
        #         os.remove(path + file)

    def get_item(self, index):
        if getattr(self, "lazy", False):
            protein = data.Protein.from_pdb(self.pdb_files[index], self.kwargs)
        else:
            protein = self.data[index].clone()
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        item = {"graph": protein}
        if self.transform:
            item = self.transform(item)
        return item

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))
