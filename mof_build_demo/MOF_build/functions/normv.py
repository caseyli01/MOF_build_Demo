import numpy as np

def normalize_vector(v):
    norm_v = v / np.linalg.norm(v)
    return norm_v