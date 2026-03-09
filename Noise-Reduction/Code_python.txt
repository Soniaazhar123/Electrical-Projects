import numpy as np
import scipy.signal as sig
import soundfile as sf
import pywt
import matplotlib.pyplot as plt

def bandpass_filter(audio, fs, lowcut=300, highcut=3400, order=5):
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype='band')
    filtered = sig.filtfilt(b, a, audio)
    return filtered

def wavelet_denoise(audio, wavelet='coif5', level=None, mode='soft', threshold=None):
    # Perform multilevel wavelet decomposition
    coeffs = pywt.wavedec(audio, wavelet, mode='per', level=level)
    # Estimate a universal threshold if not provided
    if threshold is None:
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(audio)))
    # Threshold detail coefficients (skip approximation)
    new_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]]
    denoised = pywt.waverec(new_coeffs, wavelet, mode='per')
    return denoised

def process_audio(input_wav, output_wav, apply_wavelet=True):
    audio, fs = sf.read(input_wav)
    # If stereo, convert to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    # Normalize
    audio = audio / np.max(np.abs(audio))

    # First apply band-pass filter (speech band)
    bp = bandpass_filter(audio, fs)
    print("Applied band-pass filter")

    if apply_wavelet:
        cleaned = wavelet_denoise(bp, wavelet='coif5')
        print("Applied wavelet denoising")
    else:
        cleaned = bp

    # Make sure lengths match
    cleaned = cleaned[:len(audio)]

    # Normalize again
    cleaned = cleaned / np.max(np.abs(cleaned))

    sf.write(output_wav, cleaned, fs)
    print(f"Saved cleaned audio to: {output_wav}")

    # Optional: plot comparison
    time = np.arange(len(audio)) / fs
    plt.figure(figsize=(12, 6))
    plt.subplot(2,1,1)
    plt.plot(time, audio)
    plt.title('Original / Noisy Audio')
    plt.subplot(2,1,2)
    plt.plot(time, cleaned)
    plt.title('Denoised Audio')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    input_file = "girl.wav"     # your noisy input
    output_file = "girl_cleaned.wav"  # output cleaned file
    process_audio(input_file, output_file, apply_wavelet=True)
