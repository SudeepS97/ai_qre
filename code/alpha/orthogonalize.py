
import numpy as np

def orthogonalize(alpha_matrix):
    # alpha_matrix shape: signals x assets
    q,_ = np.linalg.qr(alpha_matrix.T)
    return q.T
