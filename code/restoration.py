# restoration.py
import os
import numpy as np
import cv2
from scipy.signal import convolve2d

def run_restoration(S1):

    print("\n=== Stage E: Image Restoration ===")

    final_folder = "../outputs/final"
    os.makedirs(final_folder, exist_ok=True)

    interfered_images = sorted(
        [f for f in os.listdir(final_folder) if f.startswith("interfered_")]
    )

    # ------------------------------
    # Wiener Filter (Model-based)
    # ------------------------------
    def wiener_filter(img, psf, K=0.01):
        pad_psf = np.zeros_like(img, dtype=float)
        kh, kw = psf.shape
        pad_psf[:kh, :kw] = psf

        H = np.fft.fft2(pad_psf)
        G = np.fft.fft2(img)
        H_conj = np.conj(H)

        F_hat = (H_conj / (np.abs(H)**2 + K)) * G
        restored = np.abs(np.fft.ifft2(F_hat))

        return np.clip(restored, 0, 255).astype(np.uint8)

    # ------------------------------
    # Motion PSF (seed-based)
    # ------------------------------
    L = S1
    psf = np.zeros((L, L))
    psf[L//2, :] = 1.0 / L

    # ------------------------------
    # Blind Restoration
    # ------------------------------
    def blind_restoration(img, iterations=5, alpha=0.2):
        restored = img.astype(float)

        laplacian = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ])

        for _ in range(iterations):
            delta = convolve2d(restored, laplacian, mode='same', boundary='symm')
            restored = restored - alpha * delta

        return np.clip(restored, 0, 255).astype(np.uint8)

    print("Executing: Running restoration...")

    for i, filename in enumerate(interfered_images, start=1):
        img = cv2.imread(os.path.join(final_folder, filename), cv2.IMREAD_GRAYSCALE)

        wiener_img = wiener_filter(img, psf, K=0.01)
        blind_img = blind_restoration(img)

        cv2.imwrite(os.path.join(final_folder, f"wiener_{i}.png"), wiener_img)
        cv2.imwrite(os.path.join(final_folder, f"blind_{i}.png"), blind_img)

    print("Success: Stage E completed.")
