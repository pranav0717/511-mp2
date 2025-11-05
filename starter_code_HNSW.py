import faiss
import h5py
import numpy as np
import os
import requests

def evaluate_hnsw():

    # start your code here
    # download data, build index, run query

    with h5py.File("data/sift-128-euclidean.hdf5", "r") as f:
        xb = np.array(f["train"])
        xq = np.array(f["test"])

    d = xb.shape[1]
    index = faiss.IndexHNSWFlat(d, 16)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 200
    index.add(xb)

    D, I = index.search(xq[0:1], 10)
    with open("output.txt", "w") as f:
        for idx in I[0]:
            f.write(f"{int(idx)}\n")

    # write the indices of the 10 approximate nearest neighbours in output.txt, separated by new line in the same directory
    

if __name__ == "__main__":
    evaluate_hnsw()
