import numpy as np

def skew(x: np.ndarray) -> np.ndarray:
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ], dtype=float)

def leg_joint_cols_from_J(J: np.ndarray, tol: float = 1e-12):
    norms = np.linalg.norm(J, axis=0)
    return np.where(norms > tol)[0].astype(int)

def unflatten_jac_colmajor(J_flat: np.ndarray, rows: int, cols: int):
    """
    Input:
        J_flat: (rows*cols, p) = d vec(M)/dθ, column-major vec
    Output:
        T: (rows, cols, p) with T[:,:,k] = dM/dθ_k
    """
    J_flat = np.asarray(J_flat)
    p = J_flat.shape[1]
    mats = [np.reshape(J_flat[:, k], (rows, cols), order="F") for k in range(p)]
    return np.stack(mats, axis=2)
