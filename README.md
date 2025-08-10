# HKANLP: Link Prediction with Hyperspherical Embeddings and Kolmogorov–Arnold Networks

Official code for **“HKANLP: Link Prediction with Hyperspherical Embeddings and Kolmogorov–Arnold Networks.”**  
Hyperspherical embeddings (vMF prior) + KAN decoder for spectrally flexible adjacency reconstruction.

> **Note on datasets**  
> Datasets are **not included** in this repository because many are large and/or license-restricted.  
> Please **download from the official sources below** and place them under `data/<dataset_name>/`.

---

## Computing Infrastructure <a name="infrastructure"></a>

The experiments reported in the paper were run on the following stack.  

| Component | Specification |
|---|---|
| **CPU** | 2 × Intel Xeon Gold 6430 (32 cores each, 2.8 GHz) |
| **GPU** | 2 × NVIDIA A100 80 GB |
| **System Memory** | 512 GB DDR4-3200 |
| **Storage** | 2 TB NVMe SSD (Samsung PM9A3) |
| **Operating System** | Ubuntu 22.04.4 LTS, Linux 5.15 |
| **CUDA Driver** | 12.1 |
| **cuDNN** | 9.0 |
| **Python Environment** | Conda 23.7 |
| **Other Libraries** | GCC 11.4, CMake 3.29, OpenMPI 4.1 |

---

## Getting Started

### 1) Create a Conda environment (Python 3.8)
```bash
conda create --name HKANLP python=3.8
conda activate HKANLP
````

### 2) Install non-PyTorch dependencies

```bash
pip install -r requirements.txt
```

### 3) Pin NumPy version

Ensure `numpy==1.24.1` is installed (reinstall if needed):

```bash
pip install --upgrade --force-reinstall numpy==1.24.1
```

### 4) Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

### 5) Install PyTorch Geometric and extensions

```bash
pip install torch-geometric==2.6.1
pip install torch-cluster==1.6.3+pt24cu121 \
            torch-scatter==2.1.2+pt24cu121 \
            torch-sparse==0.6.18+pt24cu121 \
            torch-spline-conv==1.2.2+pt24cu121
```


## Scope (Under Review)

This is a **scoped, minimal release** to run demos.  

---

## Reproducibility

All hyperparameters are set in `args.py` (e.g., `dataset`, `hidden_dim`, `use_feature` `latent functions`, `num_epoch`, `learning_rate`).  
For the exact hyperparameter values per dataset, please refer to the **paper’s Appendix**.


## Quick Start

```bash
# Edit args in args.py or pass flags; then:
python train.py --dataset cora --seed 42
```
---

## Datasets (direct links)

> Download from the links below and place files under `data/<dataset_name>/`.
> We do not mirror or redistribute third-party datasets.

### Planetoid (Cora / Citeseer / Pubmed)

* Data files (`ind.<name>.*`):
  [https://github.com/kimiyoung/planetoid/tree/master/data](https://github.com/kimiyoung/planetoid/tree/master/data)

### Chameleon (WikipediaNetwork)

Geom-GCN directory:
  [https://github.com/graphdml-uiuc-jlu/geom-gcn/tree/master/new_data/chameleon](https://github.com/graphdml-uiuc-jlu/geom-gcn/tree/master/new_data/chameleon)

### Chameleon-filtered (heterophily benchmark)

* NPZ:
  [https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/chameleon_filtered.npz](https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/chameleon_filtered.npz)

### Minesweeper (heterophily benchmark)

* NPZ:
  [https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/minesweeper.npz](https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/minesweeper.npz)

### Cornell (WebKB)

* Raw files + splits (Geom-GCN):
  [https://github.com/graphdml-uiuc-jlu/geom-gcn/tree/master/new_data/cornell](https://github.com/graphdml-uiuc-jlu/geom-gcn/tree/master/new_data/cornell)

### OGB (large-scale)

* OGBN-Arxiv (zip):
  [https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip](https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip)
* OGBN-MAG (zip):
  [https://snap.stanford.edu/ogb/data/nodeproppred/mag.zip](https://snap.stanford.edu/ogb/data/nodeproppred/mag.zip)

### DrugVirus (for qualitative analysis)

* Portal (CSV export in site):
  [https://drugvirus.info](https://drugvirus.info)


---

## Notes

* This is an initial, scoped release. We welcome reports of inefficiencies or minor issues.
* Features not included here are planned to be released after acceptance.



## License

Code is released under the MIT License.
Datasets are subject to their respective original licenses/terms.
