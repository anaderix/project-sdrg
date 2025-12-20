
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



- __gnn_ml_train__/          # GNN model, training, evaluation, dataset generation
- __ml_post_processing__/   # Comparison of ML-SDRG vs exact SDRG observables
- __sdrg_ground_state__/    # Exact SDRG ground-state entanglement entropy
- __sdrg_X__/               # Finite-temperature SDRG-X implementation



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
source venv/bin/activate
pip install -r requirements.txt
```

> **Note:** All scripts should be executed from the repository root.
> We recommend setting:
>
> ```bash
> export PYTHONPATH=$(pwd)
> ```

---

## 🧠 Train the GNN (optional)

If you want to retrain the GNN from scratch:

```bash
cd gnn_ml_train
python generate_data_train.py   # data generation (see config.py)
python train_with_validation.py
```

---

## 🔁 Using the Pretrained GNN Model

A **pretrained GNN checkpoint** is provided and can be used out-of-the-box.

* **Checkpoint**: `gnn_ml_train/checkpoint.pt`
* **Model architecture & hyperparameters**: `gnn_ml_train/config.py`

The checkpoint is compatible **only** with the model definition specified in
`config.py`. Modifying the architecture (e.g. hidden dimensions or number of
layers) requires retraining.

The pretrained model is automatically loaded in the ML-assisted SDRG analysis
scripts (see below).

---

## 📊 Evaluate & Compare Entanglement Entropy

### Exact SDRG (ground state)

```bash
python sdrg_entropy.py
python plot_entropy.py
```

### ML-SDRG vs Exact SDRG

```bash
python ml_sdrg_entropy_ratio.py
```

### SDRG-X (finite temperature)

```bash
python sdrgX_entropy.py
python plot_entropy_T.py
```

---

## 📈 Example Results

* ML-SDRG reproduces entanglement entropy scaling obtained from exact SDRG
* SDRG-X entropy smoothly interpolates with temperature
* GNN predictions closely track exact SDRG decimation order

(See `ml_post_processing/entropy_results_*`)

---

## 📚 Citation

If you use this code in academic work, please cite:

```bibtex
@software{gnn_sdrg,
  title  = {GNN-assisted Strong Disorder Renormalization Group},
  author = {xxx},
  year   = {2025}
}
```

---

## 📝 License

MIT License
