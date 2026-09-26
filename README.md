# Molhiv - Francesco Bernardini

Link per i dettagli del task : 
    https://ogb.stanford.edu/docs/graphprop/#ogbg-mol

Obiettivo : Classificazione binaria di molecole (inibizione della replicazione dell'HIV) con Graph Neural Network.

---
## Cosa fare quando si torna a lavorare al progetto

```bat
conda activate molhiv
cd C:\Users\fbern\Documents\AI_Competitions\GNN
git status
pytest
```

Se tutti i test sono verdi, l'ambiente è a posto.

---

## 1 - Il Dataset

Ogni esempio è una molecola rappresentata come grafo :
    - Nodi -> Atomi
    - Archi -> Legami Chimici

Ogni esempio possiede un etichetta binaria che rappresenta la capacità di inibire o meno la replicazione del virus HIV.

Abbiamo quindi a che fare con una **graph-level classification**, ossia predizione si basa sull'intero grafo.
    *Nota* : Rivedere il concetto di Pooling per grafi

Dati originali sono stati presi da **MoleculeNet**; OGB li ridistribuisce con split e metrica standardizzati per modivi di comparabilità dei risultati.

### Dettagli sul Dataset

| | |
|---|---|
| Grafi | 41.127 |
| Nodi per grafo (media) | 25,5 |
| Archi per grafo (media) | 27,5 |
| Task | 1 (classificazione binaria) |
| Metrica | **ROC-AUC** |
| Split | **scaffold**, 80/10/10 → 32.901 / 4.113 / 4.113 |
| Feature per nodo | 9 (categoriche) |
| Feature per arco | 3 (categoriche) |

### Grafi

**Grafo** -> In questo caso stiamo parlando del grafo come Struttura Dati. Molte informazioni nel mondo reale non hanno una struttura a griglia regolare (come i pixel di un'immagine, gestiti in questo caso da una CNN)

    - *Nodi* -> Rappresentano le singole entità del dataset (Atomi, utenti in una rete sociale, città etc...)
    - *Archi* -> Rappresentato le relazioni o le connessioni tra queste entià.

Per elaborare questa tipologià si utilizzano le **Graph Neural Networks** -> Saranno il core del nostro progetto

*Nel Progetto* -> Un elemento del dataset è un oggetto 'Data' di PyTorch Geometric :

```
Data(edge_index=[2, 40], edge_attr=[40, 3], x=[19, 9], y=[1, 1], num_nodes=19)
```

- **edge_index = [2, 40]** - Struttura del grafo in formato **COO** -> *Coordinate List* : Matrice con 2 righe e 40 archi diretti. La prima riga contiene l'indice del nodo di partenza, la seconda del nodo di arrivo.

- **edge_attr = [40, 3]** - Rappresenta le *features degli archi*. Tensore da 40 righe, ognuna descritta da 3 valori (**Tipo di legame**, **Stereochimica**, **Coniugazione**) - Vi è anche un vincolo strutturale che impone -> 'edge_attr.shape[0] = edge_index.shape[1]' (In parole povere sono gli stessi archi).

- **x = [19, 9]** - Rappresenta le *feature dei nodi* - Matrice dove le 19 righe corrispondono ai nodi e le 9 colonne ci dicono come ogni nodo sia descritto da un vettore di 9 valori. (**Numero Atomico**, **Chiralità**, **Grado**, **Carica Formale**, **Numero di Idrogeni**, **Elettroni Radicali**, **Aromaticità**, **Appartenenza ad un Anello**). 

- **y = [1, 1]** - Valore target, label da predire. Formato '[1, num_task]'.

- **num_nodes = 19** - Numero di nodi da 19 nodi.

### Insidie nel progetto 

- **Forte Sbilanciamento** - Molecole attive sono una piccola minoranza. Per questo la metrica è ROC-AUC e non Accuracy. Usando accuracy come metrica mi basterebbe usare un modello che mi restituisca sempre *inattiva* avrebbe un accuracy del 95%.

- **Lo split è *scaffold*, non casuale** - Molecole sono raggruppate per scheletro strutturale - Ogni gruppo finisce intero in Training, Validation o Test. Test contiene quindi strutture chimiche mai viste durante il training. Generalizzazione misurata **Out of Distribution**. Divario Validation - Test sarà verosimilmente molto grande, così come la varianza tra seed diversi.

---

## 2 - Ambiente 

- Python 3.11 (conda / Miniforge), environment `molhiv`
- PyTorch 2.5.1 + CUDA 12.1, PyTorch Geometric 2.8, ogb 1.3.6



```bat
conda create -n molhiv python=3.11
conda activate molhiv
pip install -r requirements.txt
pip install -e .
pip install pytest
```

`pip install -e .` installa il pacchetto in modalità *editable*: `src/` resta dov'è e le modifiche sono attive immediatamente, senza reinstallare. È ciò che rende `import molhiv` funzionante da qualunque cartella, senza toccare `sys.path`.

Verifica che la GPU sia vista:

```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = torch.randn(1000, 1000, device="cuda")
print((x @ x).shape)   # l'ultima riga è il test vero: is_available() può mentire
```

---

## 3 - Struttura del Progetto

```
GNN/
├── README.md              questo file
├── .gitignore             cosa git deve ignorare
├── .gitattributes         normalizzazione dei fine riga
├── requirements.txt       dipendenze dirette, con versione esatta
├── pyproject.toml         struttura del pacchetto + configurazione di pytest
├── data/                  dataset scaricato (IGNORATO da git)
├── src/
│   └── molhiv/
│       ├── __init__.py
│       └── data.py        caricamento dataset, split, DataLoader
└── tests/
    └── test_data.py       verifiche sul dataset
```