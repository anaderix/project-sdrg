
# GNN-assisted Strong Disorder Renormalization Group (SDRG)

This repository contains implementations of **exact SDRG**, **SDRG-X**, and a
**Graph Neural Network (GNN) surrogate for SDRG decimation rules**, applied to
disordered long-range interacting spin chains.

The project combines:
- Physics-based SDRG algorithms
- Machine learning (GNNs via PyTorch Geometric)
- Post-processing and entanglement entropy analysis

---

## 📁 Repository Structure



- gnn_ml_train/          # GNN model, training, evaluation, dataset generation
- ml_post_processing/   # Comparison of ML-SDRG vs exact SDRG observables
- sdrg_ground_state/    # Exact SDRG ground-state entanglement entropy
- sdrg_X/               # Finite-temperature SDRG-X implementation



---

## 🔬 Physics Background

- Strong Disorder Renormalization Group (SDRG)
- Long-range interacting random spin chains
- Entanglement entropy scaling
- Finite-temperature SDRG-X

The GNN does **not reduce asymptotic complexity**, but acts as a fast surrogate
for the bond-selection rule, reducing prefactors and enabling parallelization.

---

## 🚀 Getting Started

### 1. Clone repository
```bash
git clone https://github.com/javahedi/project-sdrg.git
cd project-sdrg
````

### 2. Create environment

```bash
python -m venv venv
```

or

```bash
pip install -r requirements.txt
```

---

## 🧠 Train the GNN

```bash
cd gnn_ml_train
python generate_data_train.py
python train_with_validation.py
```


## 🔁 Using the Pretrained GNN Model

This repository includes a **pretrained GNN checkpoint** that can be used
directly, without retraining.

- **Checkpoint file**: `gnn_ml_train/checkpoint.pt`
- **Model architecture and hyperparameters**:
  defined in `gnn_ml_train/config.py`

The checkpoint corresponds exactly to the model configuration specified in
`config.py`. As long as this file is unchanged, the checkpoint can be loaded
and used for evaluation or inference.

### Load the pretrained model

```python
from model import SDRGNet
from checkpoint import load_checkpoint
from config import MODEL_CONFIG

model = SDRGNet(**MODEL_CONFIG)
load_checkpoint(model, "checkpoint.pt")
model.eval()


---

## 📊 Evaluate & Compare Entanglement Entropy

Exact SDRG:

```bash
cd sdrg_ground_state
python sdrg_entropy.py
python plot_entropy.py
```

ML vs Exact:

```bash
cd ml_post_processing
python ml_sdrg_entropy_ratio.py
```

SDRG-X (finite temperature):

```bash
cd sdrg_X
python sdrgX_entropy.py
python plot_entropy_T.py
```

---

## 🧪 Tests

```bash
cd gnn_ml_train/tests
pytest
```

---

## 📈 Example Results

* ML-SDRG reproduces entanglement entropy scaling
* SDRG-X entropy smoothly interpolates with temperature
* GNN predictions closely track exact decimation order

(See `ml_post_processing/entropy_results_*`)

---

## 📚 Citation

If you use this code in academic work, please cite:

```
@software{gnn_sdrg,
  title = {GNN-assisted Strong Disorder Renormalization Group},
  author = {Javad Vahedi},
  year = {2025}
}
```

---

## 📝 License

MIT License

````

---

## 6️⃣ LICENSE (MIT recommended)

Create `LICENSE`:

```txt
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy...
````

---

## 7️⃣ Initialize Git & Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: GNN-assisted SDRG framework"
git branch -M main
git remote add origin https://github.com/<your-username>/project-sdrg.git
git push -u origin main
```

