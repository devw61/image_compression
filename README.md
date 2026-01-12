# My Image Compression

These are simple image compression algorithms that I try to recreate. There can be improvements on both the FFT and Wavelet compression algorithms but for simplicity, I don't go too far into the implementation of the functions.

### Notes:
- I use jpeg on purpose because of lossy's ability to compress artifact images better than lossless
- For wavelet: I implement the haar algo instead of the 1 million other algos because of simplicity
- For FFT: I use the Cooley-Tookey cause N logN < N^2 * M^2

This is protected under MIT, but have fun with it if you want 