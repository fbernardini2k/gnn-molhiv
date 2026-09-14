"""Accesso al dataset ogbg-molhiv dell'Open Graph Benchmark."""

from __future__ import annotations

from pathlib import Path

import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader

NOME_DATASET = "ogbg-molhiv"

# Questo file sta in src/molhiv/data.py: due livelli sopra c'è la radice del repo.
RADICE_REPO = Path(__file__).resolve().parents[2]
CARTELLA_DATI = RADICE_REPO / "data"


def carica_dataset(root: Path | str = CARTELLA_DATI) -> PygGraphPropPredDataset:
    """Restituisce il dataset, scaricandolo al primo utilizzo.

    Il percorso è calcolato a partire dalla posizione di questo file, non dalla
    cartella da cui lanci il comando: così il dataset finisce sempre in data/
    che sia chiamato da uno script, da un test o dal REPL.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    return PygGraphPropPredDataset(name=NOME_DATASET, root=str(root))


def carica_split(dataset: PygGraphPropPredDataset):
    """Divide il dataset nei tre sottoinsiemi ufficiali (split scaffold)."""
    indici = dataset.get_idx_split()
    return (
        dataset[indici["train"]],
        dataset[indici["valid"]],
        dataset[indici["test"]],
    )


def crea_dataloader(
    sottoinsieme,
    batch_size: int = 32,
    mescola: bool = False,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    """Crea un DataLoader con mescolamento riproducibile.

    Il generatore esplicito serve a non dipendere dallo stato globale del
    random di torch: con lo stesso seed l'ordine dei batch è sempre lo stesso.
    """
    generatore = torch.Generator()
    generatore.manual_seed(seed)
    return DataLoader(
        sottoinsieme,
        batch_size=batch_size,
        shuffle=mescola,
        generator=generatore,
        num_workers=num_workers,
    )