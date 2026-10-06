# GNN HIV Activity Prediction

A Graph Neural Network project for predicting whether a molecule is active against HIV using molecular graph representations.

The project converts molecular structures from SMILES into graph representations and uses a PyTorch Geometric GNN for graph-level binary classification. RDKit and DeepChem are used for molecular processing and featurization, while MLflow is used for experiment tracking and model logging.

> **Project status:** Research / experimental project

## Overview

Molecules can naturally be represented as graphs:

* **Nodes** represent atoms
* **Edges** represent chemical bonds
* **Node features** describe atomic properties
* **Edge features** describe bond properties
* The complete molecular graph is used to predict HIV activity

The prediction target is:

```text
HIV_active = 1  → HIV active
HIV_active = 0  → HIV inactive
```

The dataset is highly imbalanced, with substantially fewer HIV-active compounds than inactive compounds. The project therefore includes a training-only random oversampling step to improve minority-class representation.

### Overall workflow

```text
SMILES
  │
  ▼
RDKit molecular structure
  │
  ▼
DeepChem graph featurization
  │
  ▼
PyTorch Geometric graph
  │
  ├── Node features
  ├── Edge features
  └── Molecular connectivity
  │
  ▼
Graph Neural Network
  │
  ├── Attention-based message passing
  ├── Graph pooling
  └── Graph-level representation
  │
  ▼
Binary classifier
  │
  ▼
HIV activity prediction
```

---

## Dataset

The project uses the HIV activity dataset containing molecular structures represented as SMILES strings and corresponding HIV activity labels.

The primary columns are:

| Column       | Description                           |
| ------------ | ------------------------------------- |
| `smiles`     | SMILES representation of the molecule |
| `HIV_active` | Binary HIV activity label             |

The dataset contains approximately:

* **39,684 inactive compounds**
* **1,443 active compounds**

This substantial class imbalance is an important consideration for both training and evaluation.

### Train/test split

The intended workflow is:

```text
Original dataset
       │
       ▼
 Stratified 80/20 split
       │
       ├───────────────┐
       ▼               ▼
   Training          Test
       │               │
       ▼               │
 Oversampling          │
       │               │
       ▼               ▼
Balanced training   Unmodified test
       │               │
       └───────┬───────┘
               ▼
          Model evaluation
```

Oversampling is applied **only to the training set** to prevent duplicated minority-class observations from appearing in the test set.

The current implementation uses random oversampling rather than SMOTE.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/poloarol/gnn-hiv-project.git
cd gnn-hiv-project
```

Create and activate a Python virtual environment:

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The project uses several libraries from the molecular machine-learning ecosystem, including:

* PyTorch
* PyTorch Geometric
* DeepChem
* RDKit
* scikit-learn
* MLflow
* Mango
* Pandas
* NumPy
* Matplotlib
* Seaborn

---

# Usage

The project is organized as a sequence of data preparation, graph construction, and model training steps.

## 1. Prepare the dataset

The expected raw dataset files are:

```text
data/
├── HIV_train.csv
└── HIV_test.csv
```

Each CSV should contain at least:

```text
smiles,HIV_active
```

The training and test files are generated according to the project's data-splitting workflow.

---

## 2. Split and oversample the training data

Run:

```bash
python src/oversample.py
```

This performs the training/test preparation and applies random oversampling to the minority HIV-active class.

**Important:** the test set is not oversampled.

---

## 3. Build molecular graph datasets

The raw SMILES data must be converted into PyTorch Geometric graph objects before training.

Run:

```bash
python src/dataset.py --root data --filename HIV_train.csv
```

Then process the test set:

```bash
python src/dataset.py --root data --filename HIV_test.csv
```

The dataset processor:

1. Reads the CSV file.
2. Parses each SMILES string with RDKit.
3. Removes invalid molecules.
4. Featurizes valid molecules with DeepChem's `MolGraphConvFeaturizer`.
5. Includes edge features and chirality.
6. Converts the resulting graph to a PyTorch Geometric `Data` object.
7. Stores the HIV activity label.
8. Stores the original SMILES string.
9. Stores the original CSV index.
10. Writes the processed graph to disk.

The processed graphs are stored under:

```text
data/processed/
```

The processor creates separate completion markers for training and test datasets so that PyTorch Geometric can determine whether processing has already been completed.

---

# Model

The model is implemented in:

```text
src/model.py
```

The GNN operates directly on the molecular graph.

The model receives:

```text
Node features
Edge features
Edge connectivity
Batch indices
```

and produces a graph-level prediction.

Conceptually:

```text
Molecular graph
      │
      ▼
Node + edge representations
      │
      ▼
Attention-based message passing
      │
      ▼
Graph pooling
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

This allows the model to learn molecular representations from graph structure rather than relying exclusively on manually engineered molecular descriptors.

---

# Training

Training is implemented in:

```text
src/train.py
```

The training pipeline:

* Detects GPU availability
* Loads the processed molecular graphs
* Creates PyTorch Geometric data loaders
* Initializes the GNN
* Uses weighted binary cross-entropy
* Optimizes the model with SGD
* Applies exponential learning-rate decay
* Evaluates the model periodically
* Uses early stopping
* Logs metrics and model artifacts to MLflow

The positive-class weight can be adjusted to control the relative importance of correctly identifying HIV-active compounds.

The training loop runs for up to 300 epochs, with evaluation every five epochs and early stopping when the test loss stops improving.

---

# Hyperparameter Optimization

The project uses **Bayesian hyperparameter optimization** through Mango.

The search includes parameters such as:

* Batch size
* Learning rate
* Weight decay
* SGD momentum
* Learning-rate scheduler
* Positive-class weighting
* GNN embedding dimension
* Number of attention heads
* Number of GNN layers
* Dropout
* Graph pooling ratio
* Pooling frequency
* Dense-layer dimensions

The training script performs a 100-iteration Bayesian optimization search.

The optimization objective is based on model loss.

---

# Evaluation

The model tracks several classification metrics:

* Accuracy
* Precision
* Recall
* F1 score
* ROC-AUC
* Confusion matrix

Because the HIV dataset is highly imbalanced, accuracy should not be considered in isolation.

Precision and recall are particularly important when evaluating the model's ability to identify the minority HIV-active class.

Confusion matrices are generated during evaluation and saved under:

```text
data/images/
```

They are also logged to MLflow.

---

# Experiment Tracking with MLflow

The training pipeline uses MLflow to track experiments.

Start the MLflow tracking server in one terminal:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Then, in a second terminal, run:

```bash
python src/train.py
```

The training script connects to:

```text
http://localhost:5000
```

You can then open the MLflow interface in your browser:

```text
http://localhost:5000
```

The experiments record information including:

* Hyperparameters
* Training loss
* Test loss
* Precision
* Recall
* ROC-AUC
* Number of model parameters
* Confusion matrices
* Model artifacts
* Python/library and hardware information

The trained PyTorch model is also logged to MLflow together with an input/output model signature.

---

# End-to-End Workflow

After installation, the complete workflow is:

### Step 1 — Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 2 — Prepare training/test data

```bash
python src/oversample.py
```

### Step 3 — Build training graphs

```bash
python src/dataset.py --root data --filename HIV_train.csv
```

### Step 4 — Build test graphs

```bash
python src/dataset.py --root data --filename HIV_test.csv
```

### Step 5 — Start MLflow

```bash
mlflow server --host 127.0.0.1 --port 5000
```

### Step 6 — Train and optimize the GNN

In a second terminal:

```bash
python src/train.py
```

### Step 7 — Monitor experiments

Open:

```text
http://localhost:5000
```

and inspect the tracked training runs, metrics, parameters, and model artifacts.

---

# Repository Structure

```text
gnn-hiv-project/
│
├── data/
│   ├── HIV_train.csv
│   ├── HIV_test.csv
│   ├── processed/
│   └── images/
│
├── notebooks/
│   └── Exploratory analysis and experiments
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── oversample.py
│   └── train.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# Technical Stack

| Component                   | Technology           |
| --------------------------- | -------------------- |
| Programming                 | Python               |
| Molecular chemistry         | RDKit                |
| Molecular featurization     | DeepChem             |
| Graph deep learning         | PyTorch Geometric    |
| Deep learning               | PyTorch              |
| Hyperparameter optimization | Mango                |
| Experiment tracking         | MLflow               |
| Data manipulation           | Pandas / NumPy       |
| Evaluation                  | scikit-learn         |
| Visualization               | Matplotlib / Seaborn |

---

# Why Graph Neural Networks?

Traditional molecular machine-learning workflows often represent molecules using manually engineered descriptors or fixed fingerprints.

Graph neural networks provide an alternative by learning representations directly from the molecular graph.

The model can iteratively aggregate information from neighboring atoms and chemical bonds:

```text
Atom features
     +
Bond features
     +
Molecular topology
     │
     ▼
Message passing
     │
     ▼
Learned atom representations
     │
     ▼
Graph-level pooling
     │
     ▼
Learned molecular representation
     │
     ▼
HIV activity prediction
```

This makes the project an exploration of **representation learning for molecular machine learning**.

---

# Future Directions

Potential extensions include:

* Compare GNN representations with molecular fingerprints and classical machine-learning models.
* Investigate scaffold-based train/test splitting.
* Compare different GNN architectures such as GCN, GAT and GIN.
* Improve handling of molecular class imbalance.
* Visualize learned molecular embeddings.
* Investigate attention and node importance.
* Introduce self-supervised graph pretraining.
* Explore contrastive learning for molecular representation learning.
* Combine graph representations with SMILES-based molecular language models.
* Investigate multimodal molecular representations.

---

# Project Context

This project was developed to explore graph neural networks and representation learning for molecular property prediction.

It provides practical experience with:

* Molecular graph construction
* Graph neural networks
* Attention-based message passing
* Hierarchical graph pooling
* Molecular representation learning
* Class-imbalanced classification
* Bayesian hyperparameter optimization
* Experiment tracking
* Model logging and reproducibility

The project also provides a foundation for exploring more advanced approaches to molecular representation learning, including **self-supervised learning, contrastive learning, and multimodal molecular models**.

---

# License

This project is provided for research and educational purposes.
