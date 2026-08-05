import pytest as _pytest_quarantine

_pytest_quarantine.skip(
    "QUARANTINED 2026-08-05: ImportError: cannot import name 'GraphDataset' from 'alphagrad.vertexgame' (/Users/assmuth/dsnn/alphagrad/src/alphagrad/vertexgame/__init__.py) "
    "The code under test no longer exists; kept for provenance so the suite "
    "can serve as a green/red gate. Delete or restore deliberately.",
    allow_module_level=True,
)

from torch.utils.data import DataLoader
from alphagrad.vertexgame import GraphDataset, read

import time

train_dataset = GraphDataset("./src/alphagrad/data/samples")
train_dataloader = DataLoader(train_dataset, 
                            batch_size=8, 
                            shuffle=False,
                            num_workers=2)

st = time.time()
edges = next(iter(train_dataloader))
print(edges, edges.shape)
print(time.time() - st)

# print(read("./samples/comp_graph_examples-0.hdf5", [1,2,4,5]))

