# Directed Cause-Effect Relation Extraction: A Comparison of Naive Bayes, C-GCN, and R-BERT

3-class directed Cause-Effect relation extraction.
Three models (Naive Bayes, R-BERT, C-GCN) evaluated on two datasets (SemEval-2010 Task 8 and Causal News Corpus).

---

## Infrastructure

- **GPU**: NVIDIA RTX 1000 Ada Generation Laptop GPU, 6 GB VRAM
- **CUDA**: 12.4
- **PyTorch**: 2.6.0
- **Python**: 3.11.9
- Naive Bayes runs on CPU. R-BERT and C-GCN require a CUDA-capable GPU.

---

## Reproducibility note

The original experiments were run natively on Windows (NVIDIA RTX 1000, CUDA 12.4). Docker, even on Windows, runs inside a Linux environment via WSL2. PyTorch uses cuDNN internally for GPU operations, and cuDNN behaves differently between native Windows and Linux. This means that even with identical random seeds, results produced inside Docker may differ from the pre-computed results stored in `results/`.

- **To verify the code runs**: use Docker.
- **To reproduce the reported results**: re-run locally on the same hardware (Windows, NVIDIA RTX 1000 Ada, CUDA 12.4).
- **To inspect results without re-running**: all results are pre-computed in `results/`.

Naive Bayes is not affected, it runs on CPU and is fully reproducible across platforms.

---

## Pre-computed results

All results are already included in `results/`. Each model/dataset folder contains:
- `metrics.json`: macro F1, per-class precision/recall/F1
- `report.txt`: human readable report
- `predictions.txt`: predicted labels for the test set
- `confusion_matrix.png`: confusion matrix plot

R-BERT and C-GCN models were run with 5 seeds. The top-level folder contains the initial run (seed 77 for R-BERT, seed 1234 for C-GCN) and `seed_1/` through `seed_4/` contain the remaining runs. Thus, no need to retrain the models to inspect the results.

---

## Setup — Docker

### Requirements
- NVIDIA GPU with CUDA 12.4+ drivers
- On Linux: Docker + NVIDIA Container Toolkit
- On Windows: Docker Desktop with WSL2 backend

### 1. Build the image

From the repo root:

```bash
docker build -t causal-extraction .
```

Installs all dependencies. On first run this may take several minutes.

### 2. Start Jupyter

```bash
docker run --gpus all -p 8888:8888 causal-extraction
```

Then open `http://localhost:8888` in your browser. All notebooks are in the `notebooks/` folder. 

See "Running the notebooks" below for the full list of notebooks and how to use them.

---

## Setup — Local (without Docker)

### Requirements
- Python 3.11
- NVIDIA GPU with CUDA 12.4 drivers

### 1. Install dependencies

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
pip install jupyter
```

### 2. Start Jupyter

```bash
jupyter notebook
```

See "Running the notebooks" below for the full list of notebooks and how to use them.

---

## Running the notebooks

Open any notebook from the `notebooks/` folder and run all cells. Each notebook is self-contained and includes explanations.

| Notebook | Model | Dataset |
|---|---|---|
| `naive_bayes_semeval.ipynb` | Naive Bayes | SemEval-2010 |
| `naive_bayes_cnc.ipynb` | Naive Bayes | CNC |
| `r_bert_semeval.ipynb` | R-BERT | SemEval-2010 |
| `r_bert_cnc.ipynb` | R-BERT | CNC |
| `cgcn_semeval.ipynb` | C-GCN | SemEval-2010 |
| `cgcn_cnc.ipynb` | C-GCN | CNC |
| `reproduction/naive_bayes_19class_semeval.ipynb` | Naive Bayes (19-class) | SemEval-2010 |

Results are written to the corresponding `results/` subfolder after each run.

### Error analysis

After running all models, run the error analysis scripts. 

From the repo root (local):

```bash
python notebooks/error_analysis_SemEvalData.py
python notebooks/error_analysis_CNCData.py
```

With Docker:

```bash
docker run --gpus all causal-extraction python notebooks/error_analysis_SemEvalData.py
docker run --gpus all causal-extraction python notebooks/error_analysis_CNCData.py
```
