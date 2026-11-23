"""Index builder for the first version."""
import numpy as np
import faiss
import json
import os

DATA_DIR = "data"

# load data
vecs = np.load(os.path.join(DATA_DIR, "recipe_vectors.npy"))
recipe_ids = np.load(os.path.join(DATA_DIR, "recipe_ids.npy"), allow_pickle=True)
# optional: load metadata mapping
with open(os.path.join(DATA_DIR, "recipes_for_embed.csv"), "r", encoding="utf-8") as f:
    pass

# ensure float32
vecs = vecs.astype("float32")
N, D = vecs.shape
print("N,D =", N, D)

# normalize if using inner-product / cosine
faiss.normalize_L2(vecs)

# build index (IVF not necessary for small N). Here we use IndexFlatIP (cosine after normalize)
index = faiss.IndexFlatIP(D)
index.add(vecs)
print("Added", index.ntotal, "vectors")

# save index
faiss.write_index(index, os.path.join(DATA_DIR, "recipe_index.faiss"))
np.save(os.path.join(DATA_DIR, "recipe_ids_clean.npy"), recipe_ids)
print("Saved index and ids")
