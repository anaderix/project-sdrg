import json
import numpy as np

with open("/home/javad/Desktop/project-sdrg/data/realization_000/step_000.json") as f:
    sample = json.load(f)

edge_features = np.array(sample["edge_features"])
target = sample["target_edge"]

print("Target logJ:", edge_features[target][0])
print("Max logJ:", edge_features[:,0].max())

num_edges = len(sample["edge_features"])
print("Edges:", num_edges)
print("Positive fraction:", 1 / num_edges)
