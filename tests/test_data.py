"""Verifiche sul caricamento di ogbg-molhiv.

Se un giorno il download si corrompesse, OGB cambiasse versione del dataset
o qualcuno toccasse data.py, ce ne accorgiamo qui — non tre settimane dopo,
davanti a risultati inspiegabili.
"""

import pytest

from molhiv.data import carica_dataset, carica_split, crea_dataloader

# Valori ufficiali: https://ogb.stanford.edu/docs/graphprop/#ogbg-mol
N_GRAFI = 41_127
N_TASK = 1
N_FEATURE_NODO = 9
N_FEATURE_ARCO = 3
DIM_SPLIT = (32_901, 4_113, 4_113)


@pytest.fixture(scope="module")
def dataset():
    """Caricato una volta sola per tutto il file: il download è lento."""
    return carica_dataset()


def test_numero_di_grafi(dataset):
    assert len(dataset) == N_GRAFI


def test_numero_di_task(dataset):
    assert dataset.num_tasks == N_TASK


def test_dimensioni_delle_feature(dataset):
    grafo = dataset[0]
    assert grafo.x.shape[1] == N_FEATURE_NODO
    assert grafo.edge_attr.shape[1] == N_FEATURE_ARCO


def test_struttura_degli_archi(dataset):
    """edge_index è COO e ogni arco ha i suoi attributi."""
    grafo = dataset[0]
    assert grafo.edge_index.shape[0] == 2
    assert grafo.edge_index.shape[1] == grafo.edge_attr.shape[0]
    # nessun arco può puntare a un nodo che non esiste
    assert int(grafo.edge_index.max()) < grafo.num_nodes


def test_split_ufficiali(dataset):
    train, valid, test = carica_split(dataset)
    assert (len(train), len(valid), len(test)) == DIM_SPLIT
    assert len(train) + len(valid) + len(test) == N_GRAFI


def test_dataloader_conserva_i_grafi(dataset):
    """Senza mescolamento i grafi escono nell'ordine e nella forma originali."""
    sottoinsieme = dataset[:64]
    caricatore = crea_dataloader(sottoinsieme, batch_size=16, mescola=False)

    nodi_dal_loader = [
        int(n) for batch in caricatore for n in batch.batch.bincount()
    ]
    nodi_attesi = [int(sottoinsieme[i].num_nodes) for i in range(64)]

    assert nodi_dal_loader == nodi_attesi


def test_mescolamento_riproducibile(dataset):
    """Stesso seed, stesso ordine: è la base di tutta la riproducibilità."""
    sottoinsieme = dataset[:200]

    def ordine(seed: int) -> list[int]:
        caricatore = crea_dataloader(sottoinsieme, batch_size=16, mescola=True, seed=seed)
        return [int(n) for batch in caricatore for n in batch.batch.bincount()]

    assert ordine(0) == ordine(0)