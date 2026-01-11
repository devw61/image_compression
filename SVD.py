import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (16, 8)


def svd(A):
    m, n = A.shape
    r = min(m, n)

    Atrans_A = A.T @ A
    eigvals, V = np.linalg.eig(Atrans_A)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    V = V[:, idx]

    sing_vals = np.sqrt(np.maximum(eigvals, 0))[:r]

    Sigma = np.diag(sing_vals)
    U = np.zeros((m, r))

    for i in range(r):
        if sing_vals[i] > 1e-10:
            U[:, i] = (A @ V[:, i]) / sing_vals[i]

    return U, Sigma, V[:, :r].T


def run_svd_img_comp():
    A = imread('image.png')
    X = np.mean(A, -1)  # convert to grayscale

    img = plt.imshow(X, cmap='gray')
    plt.axis('off')
    plt.savefig("before.png", dpi=300, bbox_inches='tight', pad_inches=0)

    U, S, VT = svd(X)
    U_, S_, VT_ = np.linalg.svd(X, full_matrices=False)
    S_ = np.diag(S_)

    j = 0
    for r in (5, 20, 50, 100):
        X_approx = U[:, :r] @ S[:r, :r] @ VT[:r, :]
        X_approx_ = U_[:, :r] @ S_[:r, :r] @ VT_[:r, :]

        plt.figure(j+1)
        j += 1

        img = plt.imshow(X_approx, cmap='gray')
        plt.axis('off')
        plt.savefig(f"svd_{r}.png", dpi=300, bbox_inches='tight', pad_inches=0)
        plt.savefig(f"svd_numpy_{r}.png", dpi=300, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    run_svd_img_comp()