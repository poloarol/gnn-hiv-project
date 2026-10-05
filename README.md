# GNN HIV Activity Prediction

A Graph Neural Network project for predicting whether a molecule is active against HIV using molecular graph representations.
The project uses PyTorch Geometric to represent molecules as graphs, with RDKit and DeepChem used for molecular processing and featurization. The project also incorporates MLflow for experiment tracking and reproducibility.
Project status: Work in progress / research project

Overview
The goal of this project is to build a Graph Neural Network (GNN) capable of predicting HIV activity directly from molecular structures.
Each molecule is represented as a graph:

- Nodes represent atoms.
- Edges represent chemical bonds.
- Node features describe the properties of individual atoms.
- Edge features describe the properties of chemical bonds.
- The model performs graph-level binary classification.

The target variable is:
- HIV_active

where:
1 = HIV active
0 = HIV inactive

The project focuses particularly on the severe class imbalance present in HIV activity data and investigates oversampling as a way of improving minority-class representation during training.

## Current Project Status
The following components are currently being developed or implemented:
 - HIV dataset exploration
 - Class distribution analysis
 - Molecular visualization with RDKit
 - Python virtual environment setup
 - Molecular graph processing with PyTorch Geometric
 - Integration with RDKit
 - Integration with DeepChem
 - Training/test data splitting
 - Stratified 80/20 train/test split
 - Random oversampling of the minority class
 - MLflow experiment tracking setup
 - Final GNN architecture
 - Systematic hyperparameter optimization
 - Final model evaluation
 - Model comparison
 - Final experiment results
 - Model checkpointing / deployment

Dataset
The project uses the HIV activity dataset containing molecular structures represented by SMILES strings and corresponding HIV activity labels.
The primary columns used by the project include:

Column	Description
- smiles	SMILES representation of the molecule
- HIV_active	Binary HIV activity label

The dataset is highly imbalanced, with substantially fewer HIV-active compounds than inactive compounds.

This imbalance is an important consideration for both model training and evaluation.

Class imbalance

The project first examines the distribution of:

hiv_df["HIV_active"].value_counts()


The class distribution is visualized using a logarithmic count plot to make the difference between the classes easier to see.

Data Splitting

The dataset is split into training and test sets using an 80/20 split.

The split is stratified by HIV_active so that the class proportions are approximately preserved between the two datasets.

train_data, test_data = train_test_split(
    data,
    test_size=0.20,
    stratify=data["HIV_active"],
    random_state=42
)

Important: oversampling is applied only to the training data

The intended workflow is:
```
Original dataset
       │
       ▼
  80/20 stratified split
       │
       ├───────────────┐
       ▼               ▼
    Training          Test
       │               │
       ▼               │
  Oversampling         │
       │               │
       ▼               ▼
Balanced training   Unmodified test
       │               │
       └───────┬───────┘
               ▼
          Model evaluation
```

This prevents duplicated training examples from appearing in the test set and causing data leakage.

Oversampling

Because the HIV-active class is substantially smaller than the inactive class, the project currently uses random oversampling.

Positive examples are sampled with replacement until the two classes are balanced in the training set.

Conceptually:
```
additional_pos = positive_data.sample(
    n=neg_class - pos_class,
    replace=True,
    random_state=42
)
```

The resulting training set contains approximately equal numbers of active and inactive molecules.

Random oversampling vs. SMOTE

The current implementation uses random oversampling, not SMOTE.

Random oversampling duplicates existing molecular examples.

SMOTE, by contrast, creates synthetic feature-space samples by interpolating between minority examples. Because molecular data is represented as graphs rather than ordinary tabular feature vectors, applying conventional SMOTE directly to molecular graphs requires additional consideration.

Molecular Representation

Molecules are converted from SMILES strings into graph representations using the molecular-processing tools in the project.

A simplified representation is:
```
                 Molecular Graph

              O
              |
        C ─── C ─── N
        │
        C
       / \
      C   C

```
Each graph contains:

Nodes  → atoms
Edges  → chemical bonds
Graph  → complete molecule
Label  → HIV activity


RDKit is used to parse and inspect molecular structures, while PyTorch Geometric provides the graph data structures used by the GNN.

Molecular Visualization

RDKit is used to visualize molecules from their SMILES representations.

For example:
```
from rdkit import Chem
from rdkit.Chem import Draw

mol = Chem.MolFromSmiles(smiles)

Draw.MolToImage(mol)
```

The project also supports displaying multiple randomly selected active and inactive molecules for qualitative inspection.

Model

The central model is a Graph Neural Network implemented with PyTorch Geometric.

The general workflow is:
```
SMILES
  │
  ▼
RDKit molecular representation
  │
  ▼
Molecular graph
  │
  ├── Node features
  │
  └── Edge features
  │
  ▼
Graph Neural Network
  │
  ▼
Graph-level representation
  │
  ▼
Binary classifier
  │
  ▼
P(HIV active)
```

The final GNN architecture and hyperparameters are still under development.

Experiment Tracking

The project uses MLflow to track experiments.

The environment information being recorded includes:

PyTorch version

PyTorch Geometric version

DeepChem version

RDKit version

NumPy version

pandas version

CUDA availability

CUDA version

GPU device name

For example:

mlflow.log_params({
    "torch_version": torch.__version__,
    "torch_geometric_version": torch_geometric.__version__,
    "deepchem_version": dc.__version__,
    "rdkit_version": rdkit.__version__,
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda or "N/A",
    "gpu_name": (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "N/A"
    ),
})


The intention is to make model comparisons reproducible by recording both model hyperparameters and the software/hardware environment used for each run.

Project Structure

The repository is currently organized approximately as follows:
```
gnn-hiv-project/
│
├── data/
│   └── ...
│
├── notebooks/
│   └── ...
│
├── src/
│   └── ...
│
├── .gitignore
└── README.md

data/
```
Contains the project datasets and generated data files.

Raw and processed datasets should be kept separate where possible.

notebooks/

Contains exploratory analysis and experimentation, including:

Dataset exploration

Class distribution analysis

Molecular visualization

Model experimentation

src/

Contains reusable Python code for the project, including data processing and training-related scripts.

Environment

The project uses a Python virtual environment.

Because the project depends on packages such as PyTorch, PyTorch Geometric, RDKit, and DeepChem, Python/package compatibility is important.

A Python 3.12 environment is currently recommended.

Using uv:

uv python install 3.12

uv venv --python 3.12 env


Activate the environment on Windows:

.\env\Scripts\Activate.ps1


Then install the required packages.

Jupyter

To use the environment as a Jupyter kernel:

uv pip install ipykernel


Register the environment:
```
python -m ipykernel install \
    --user \
    --name=gnn-hiv \
    --display-name "Python (gnn-hiv)"
```

The Python (gnn-hiv) kernel can then be selected from Jupyter.

Reproducibility

Random seeds are used where appropriate to make experiments reproducible.

For example:

random_state=42


is currently used for dataset splitting and oversampling.

MLflow is also used to record the software and hardware environment associated with experiments.

Future versions of the project will additionally track:

Model hyperparameters

Dataset versions

Training metrics

Validation metrics

Test metrics

Model checkpoints

Git commit

Random seeds

Evaluation

Because this is an imbalanced binary classification problem, accuracy alone is not sufficient to evaluate the model.

The final evaluation should include metrics such as:

ROC-AUC

PR-AUC / Average Precision

Precision

Recall

F1 score

Confusion matrix

Particular attention should be given to precision-recall performance, since the positive HIV-active class is relatively rare.

Final model results will be added once the GNN training pipeline and experiments have been finalized.

Future Work

The next stages of the project are:

Finalize the GNN architecture.

Establish a baseline model.

Train using the imbalanced dataset.

Train using random oversampling.

Compare model performance with and without oversampling.

Perform systematic hyperparameter optimization.

Track all experiments using MLflow.

Evaluate using ROC-AUC and PR-AUC alongside precision, recall, and F1.

Investigate molecular/scaffold-based splitting to provide a more stringent measure of generalization.

Save and document the best-performing model.

Limitations

The current project is still under development.

In particular:

The final model architecture has not yet been established.

Final benchmark results have not yet been reported.

Random oversampling is currently being used rather than synthetic graph generation.

A random 80/20 split may provide an optimistic estimate of generalization because chemically similar molecules can occur in both training and test sets.

A scaffold-based split is therefore an important future improvement for evaluating whether the model generalizes to structurally different molecules.

Technologies

The project currently uses:

Python

PyTorch

PyTorch Geometric

RDKit

DeepChem

pandas

NumPy

Jupyter

MLflow

Project Status

This repository represents an ongoing exploration of graph-based molecular machine learning for HIV activity prediction.

The current priority is establishing a reproducible baseline, handling the severe class imbalance appropriately, and comparing GNN experiments using consistent evaluation and MLflow tracking.
