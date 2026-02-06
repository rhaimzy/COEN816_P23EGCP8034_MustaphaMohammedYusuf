# filters.py
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def run_filters(seed, S1):

    print("\n=== Stage B–D: Filtering Pipeline ===")

    # ✅ Correct project-root paths
    corrupt_folder = "../outputs/corrupted"
    intermediate_folder = "../outputs/intermediate"
    final_folder = "../outputs/final"

    os.makedirs(intermediate_folder, exist_ok=True)
    os.makedirs(final_folder, exist_ok=True)

    # Toggle visualization
    show_plots = True

    # ------------------------------
    # Stage B: Histogram Enhancement
    # ------------------------------

    def manual_histogram_equalization(img):
        hist = np.zeros(256, dtype=int)
        for p in img.ravel():
            hist[p] += 1

        cdf = hist.cumsum()
        cdf_norm = cdf / cdf[-1]

        mapping = np.floor(255 * cdf_norm).astype(np.uint8)
        return mapping[img]

    def local_adaptive_enhancement(img, window=15):
        pad = window // 2
        padded = np.pad(img, pad, mode='reflect')
        enhanced = np.zeros_like(img)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                local = padded[i:i+window, j:j+window]
                local_mean = np.mean(local)
                enhanced[i, j] = np.clip(img[i, j] + (img[i, j] - local_mean), 0, 255)

        return enhanced.astype(np.uint8)

    noisy_images = sorted(os.listdir(corrupt_folder))

    print("Executing: Stage B")

    for i, filename in enumerate(noisy_images, start=1):
        img = cv2.imread(os.path.join(corrupt_folder, filename), cv2.IMREAD_GRAYSCALE)

        global_enhanced = manual_histogram_equalization(img)
        local_enhanced = local_adaptive_enhancement(img)

        cv2.imwrite(os.path.join(intermediate_folder, f"global_{i}.png"), global_enhanced)
        cv2.imwrite(os.path.join(intermediate_folder, f"local_{i}.png"), local_enhanced)

        if show_plots:
            plt.figure(figsize=(12,4))

            plt.subplot(1,3,1)
            plt.imshow(img, cmap='gray')
            plt.title("Noisy")
            plt.axis('off')

            plt.subplot(1,3,2)
            plt.imshow(global_enhanced, cmap='gray')
            plt.title("Global EQ")
            plt.axis('off')

            plt.subplot(1,3,3)
            plt.imshow(local_enhanced, cmap='gray')
            plt.title("Local Adaptive")
            plt.axis('off')

            plt.show()

    print("Success: Stage B completed.")

    # ------------------------------
    # Stage C: Spatial Filtering
    # ------------------------------

    def separable_gaussian_filter(img, sigma=1.0):
        size = 5
        ax = np.arange(-(size//2), size//2 + 1)
        kernel = np.exp(-0.5 * (ax / sigma) ** 2)
        kernel /= kernel.sum()

        temp = convolve2d(img, kernel.reshape(1, -1), mode='same', boundary='symm')
        smooth = convolve2d(temp, kernel.reshape(-1, 1), mode='same', boundary='symm')

        return np.clip(smooth, 0, 255).astype(np.uint8)

    def median_filter(img, k=5):
        pad = k // 2
        padded = np.pad(img, pad, mode='edge')
        filtered = np.zeros_like(img)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                neighborhood = padded[i:i+k, j:j+k]
                filtered[i, j] = np.median(neighborhood)

        return filtered

    def motion_blur(img, length):
        kernel = np.zeros((length, length))
        kernel[length//2, :] = 1.0 / length
        blurred = convolve2d(img, kernel, mode='same', boundary='symm')
        return np.clip(blurred, 0, 255).astype(np.uint8), kernel

    def inverse_filter(img, kernel, eps=1e-3):
        pad_kernel = np.zeros_like(img, dtype=float)
        kh, kw = kernel.shape
        pad_kernel[:kh, :kw] = kernel

        H = np.fft.fft2(pad_kernel)
        G = np.fft.fft2(img)

        H_inv = np.conj(H) / (np.abs(H)**2 + eps)
        F_hat = H_inv * G

        restored = np.abs(np.fft.ifft2(F_hat))
        return np.clip(restored, 0, 255).astype(np.uint8)

    enhanced_images = sorted([f for f in os.listdir(intermediate_folder) if f.startswith("global_")])

    print("Executing: Stage C")

    for i, filename in enumerate(enhanced_images, start=1):
        img = cv2.imread(os.path.join(intermediate_folder, filename), cv2.IMREAD_GRAYSCALE)

        gaussian_img = separable_gaussian_filter(img)
        median_img = median_filter(img)

        L = S1 + (i % 3)
        blurred_img, psf = motion_blur(img, L)
        restored_img = inverse_filter(blurred_img, psf)

        cv2.imwrite(os.path.join(final_folder, f"gaussian_{i}.png"), gaussian_img)
        cv2.imwrite(os.path.join(final_folder, f"median_{i}.png"), median_img)
        cv2.imwrite(os.path.join(final_folder, f"blurred_{i}.png"), blurred_img)
        cv2.imwrite(os.path.join(final_folder, f"restored_{i}.png"), restored_img)

    print("Success: Stage C completed.")

    # ------------------------------
    # Stage D: Frequency Filtering
    # ------------------------------

    def inject_periodic_interference(img, i, seed):
        M, N = img.shape
        F = fftshift(fft2(img))

        f0 = ((seed + i) % 16) - 8
        A = 2 + (i % 7)

        for x in range(M):
            for y in range(N):
                F[x, y] += A * np.exp(1j * 2 * np.pi * f0 * y / N)

        interfered = np.abs(ifft2(ifftshift(F)))
        return np.clip(interfered, 0, 255).astype(np.uint8), F

    def adaptive_notch_filter(F, u0, v0, D0=5):
        M, N = F.shape
        U, V = np.meshgrid(np.arange(N), np.arange(M))
        U -= N//2
        V -= M//2

        H = 1 - np.exp(-0.5 * ((U-u0)**2 + (V-v0)**2) / D0**2)
        H *= 1 - np.exp(-0.5 * ((U+u0)**2 + (V+v0)**2) / D0**2)

        return F * H

    restored_images = sorted([f for f in os.listdir(final_folder) if f.startswith("restored_")])

    print("Executing: Stage D")

    for i, filename in enumerate(restored_images, start=1):
        img = cv2.imread(os.path.join(final_folder, filename), cv2.IMREAD_GRAYSCALE)

        interfered_img, F_interfered = inject_periodic_interference(img, i, seed)
        cv2.imwrite(os.path.join(final_folder, f"interfered_{i}.png"), interfered_img)

        F_notch = adaptive_notch_filter(F_interfered, 0, 5)
        notch_img = np.abs(ifft2(ifftshift(F_notch)))
        notch_img = np.clip(notch_img, 0, 255).astype(np.uint8)

        cv2.imwrite(os.path.join(final_folder, f"notch_{i}.png"), notch_img)

    print("Success: Stage D completed.")
