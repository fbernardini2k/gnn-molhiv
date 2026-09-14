# Molhiv - Francesco Bernardini

Link per i dettagli del task : 
    https://ogb.stanford.edu/docs/graphprop/#ogbg-mol

Obiettivo : Classificazione binaria di molecole (inibizione della replicazione dell'HIV) con Graph Neural Network.

## Ambiente 

- Python 3.11 (conda / Miniforge), environment `molhiv`
- PyTorch 2.5.1 + CUDA 12.1, PyTorch Geometric 2.8, ogb 1.3.6

```bat
conda create -n molhiv python=3.11
conda activate molhiv
pip install -r requirements.txt
```