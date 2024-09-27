# Audio-Compression-using-SVD

This repository contains all the resources used while researching about this project.

### Overview of the project
Convert audio signal to a matrix (spectrogram) &rarr; Apply SVD &rarr; Convert matrix back to audio signal form.

**In bit more detail:** (response from Bing - Microsoft Edge)

Singular Value Decomposition (SVD) is a powerful mathematical technique used in audio compression to reduce file sizes while maintaining sound quality. Here’s a simplified explanation of how it works:
- **Convert Audio to Spectrogram:** The audio signal is transformed into a spectrogram using the Short-Time Fourier Transform (STFT). This breaks down the audio into its time-frequency components.
- **Apply SVD:** The spectrogram is then decomposed using SVD, which separates it into three matrices: ( U ), ( \Sigma ), and ( V^T ). The matrix ( \Sigma ) contains singular values that represent the importance of each component.
- **Truncate Singular Values:** By keeping only the largest singular values and discarding the smaller ones, you can significantly reduce the amount of data. This step is crucial for compression.
- **Reconstruct Audio:** The compressed spectrogram is then used to reconstruct the audio signal. The Griffin-Lim algorithm is often employed for this purpose1.
The key advantage of using SVD for audio compression is that it allows for high compression ratios with minimal loss of audio quality. This makes it an efficient method for reducing storage requirements without significantly compromising sound fidelity1.
