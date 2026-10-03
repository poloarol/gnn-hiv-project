import argparse
import os

import numpy as np
# import mlflow
import deepchem as dc
import pandas as pd
import rdkit
import torch
import torch_geometric

from rdkit import Chem
from torch_geometric.data import Dataset
from tqdm import tqdm

print("Dataset module loaded successfully.")
print("Torch version: ", torch.__version__)
print("Torch Geometric version: ", torch_geometric.__version__)
print("DeepChem version: ", dc.__version__)
print("CUDA available: ", torch.cuda.is_available())


class MoleculeDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        
        self.test = test
        self.filename = filename
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        
        if self.test:
            return [f"data_test_{i}.pt" for i in list(self.data.index)]
        else:
            return [f"data_{i}.pt" for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True, use_partial_charge=True)
        
        for _, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol is not None:
                feat = featurizer._featurize(mol)
                graph = feat.to_pyg_graph()
                graph.y = self._get_label(row['HIV_active'])
                graph.smiles = row['smiles']
                if self.test:
                    torch.save(graph, os.path.join(self.processed_dir, f"data_test_{row['index']}.pt"))
                else:
                    torch.save(graph, os.path.join(self.processed_dir, f"data_{row['index']}.pt"))

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))        
        return data


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--filename", type=str, default="molecules.csv")
    args = parser.parse_args()
    
    dataset = MoleculeDataset(root=args.root, filename=args.filename)
    print("Number of molecules: ", len(dataset))