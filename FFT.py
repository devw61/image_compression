import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt

# Cooley-Tukey FFT implementation
def fft1d(X):
    X = np.asarray(X, dtype=complex)
    N = X.shape[0]

    if N <= 1:
        return X
    
    if N % 2 != 0:
        raise ValueError("Size of input must be a power of 2")
    
    # divide and conquer
    even = fft1d(X[0::2])
    odd = fft1d(X[1::2])

    # e ^ (-2j * pi * k / N)
    T = np.exp(-2j * np.pi * np.arange(N) / N)

    # combine even odd
    return np.concatenate([
        even + T[:N//2] * odd,
        even + T[N//2:] * odd
    ])


def fft2(A):
    A = np.asarray(A, dtype=float)

    m,n = A.shape

    F = np.zeros((m,n), dtype=complex)

    # FFT on columns
    for i in range(m):
        F[i,:] = fft1d(A[i,:])

    # FFT on rows
    for j in range(n):
        F[:,j] = fft1d(F[:,j])

    return F

def ifft1d(X):
    return np.conj(fft1d(np.conj(X))) / len(X)

def ifft2(F):
    F = np.asarray(F, dtype=complex)
    m, n = F.shape

    A = np.zeros((m,n), dtype=complex)


    # in dif order to prove they are interchangeable
    # IFFT on columns
    for j in range(n):
        A[:,j] = ifft1d(F[:,j])


    # IFFT on rows
    for i in range(m):
        A[i,:] = ifft1d(A[i,:])

    return A


def pad_to_pow2(A):
    M, N = A.shape
    M2 = 1 << (M - 1).bit_length()
    N2 = 1 << (N - 1).bit_length()

    B = np.zeros((M2, N2))
    B[:M, :N] = A
    return B


def run_fft_img_comp():
    A = imread('image.png')
    X = np.mean(A, -1)  # convert to grayscale

    m, n = X.shape

    # ensure dimensions are powers of 2
    X = pad_to_pow2(X)

    plt.imshow(X, cmap='gray')
    plt.axis('off')
    plt.savefig("before.png", dpi=300, bbox_inches='tight', pad_inches=0)

    FT_ = np.fft.fft2(X)
    FT = fft2(X)
    print(f"Validation: {np.max(np.abs(FT_ - FT))} | should be close to 0")

    ft_sort = np.sort(np.abs(FT.reshape(-1)))

    for keep in (0.01, 0.05, 0.1, 0.2):
        threshold = ft_sort[int((1-keep)*len(ft_sort))]
        ind = np.abs(FT) >= threshold
        Ftlow = FT * ind

        Flow = ifft2(Ftlow).real
        Flow_ = np.fft.ifft2(Ftlow).real

        print(f"Keep: {keep*100:4.1f}% | Reconstruction error (numpy): {np.max(np.abs(X - Flow_))} | (mine): {np.max(np.abs(X - Flow))}")

        plt.imshow(Flow, cmap='gray')
        plt.axis('off')
        plt.savefig(f"fft_keep_{int(keep*100)}.png", dpi=300, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    run_fft_img_comp()


