import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import pywt

def haar_1d(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if N > 0 and (N & (N - 1)) != 0:
        raise ValueError("Size of input must be a power of 2")

    if N <= 1:
        return x

    # averaging and differencing
    avg = (x[0::2] + x[1::2]) / np.sqrt(2)
    diff = (x[0::2] - x[1::2]) / np.sqrt(2)

    return np.concatenate([haar_1d(avg), diff])


def inv_haar_1d(x):
    n = len(x)
    
    if n <= 1:
        return x
    
    h = n // 2

    avg = inv_haar_1d(x[:h])  # Recursively reconstruct the averages
    diff = x[h:]

    out = np.zeros(n)
    out[0::2] = (avg + diff) / np.sqrt(2)
    out[1::2] = (avg - diff) / np.sqrt(2)
    return out


def haar_2d(A, levels=1):
    A = A.astype(float).copy()
    h, w = A.shape

    for level in range(levels):
        h2 = h >> level
        w2 = w >> level

        # Rows
        for i in range(h2):
            A[i, :w2] = haar_1d(A[i, :w2])

        # Columns
        for j in range(w2):
            A[:h2, j] = haar_1d(A[:h2, j])

    return A


def inv_haar_2d(A, levels=1):
    A = A.astype(float).copy()
    h, w = A.shape

    for level in reversed(range(levels)):
        h2 = h >> level
        w2 = w >> level

        # Columns
        for j in range(w2):
            A[:h2, j] = inv_haar_1d(A[:h2, j])

        # Rows
        for i in range(h2):
            A[i, :w2] = inv_haar_1d(A[i, :w2])

    return A


def threshold(A, keep_percent):
    flat = np.abs(A.flatten())
    thresh = np.percentile(flat, 100 - keep_percent)
    A[np.abs(A) < thresh] = 0
    return A


def wavelet_compress(A, levels=4, keep_percent=5):
    A = A.astype(float)

    W = haar_2d(A, levels)
    W = threshold(W, keep_percent)
    A_rec = inv_haar_2d(W, levels)

    return np.clip(A_rec, 0, 255)


def pad_to_pow2(A):
    M, N = A.shape
    M2 = 1 << (M - 1).bit_length()
    N2 = 1 << (N - 1).bit_length()

    B = np.zeros((M2, N2))
    B[:M, :N] = A
    return B


if __name__ == "__main__":
    A = imread('image.png')
    X = np.mean(A, -1)  # convert to grayscale
    
    # Scale to [0, 255] range if needed
    if X.max() <= 1.0:
        X = X * 255.0
    
    # ensure dimensions are powers of 2
    X = pad_to_pow2(X)

    plt.imshow(X, cmap='gray')
    plt.axis('off')
    plt.savefig("before.png", dpi=300, bbox_inches='tight', pad_inches=0)

    for keep in (0.5, 1, 5, 10, 20):
        X_rec = wavelet_compress(X, levels=4, keep_percent=keep)

        coeffs = pywt.wavedec2(X, 'haar', level=4)
        arr, coeff_slices = pywt.coeffs_to_array(coeffs)
        arr = threshold(arr, keep)
        coeffs_thresh = pywt.array_to_coeffs(arr, coeff_slices, output_format='wavedec2')
        X_rec = pywt.waverec2(coeffs_thresh, 'haar')

        print(f"Keep: {keep}% | Reconstruction error: (numpy) {np.max(np.abs(X - X_rec))} - (mine) {np.max(np.abs(X - X_rec))}")

        plt.imshow(X_rec, cmap='gray')
        plt.axis('off')
        plt.savefig(f"wavelet_keep_{int(keep)}.png", dpi=300, bbox_inches='tight', pad_inches=0)