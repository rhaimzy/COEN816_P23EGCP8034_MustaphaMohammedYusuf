# analysis.py
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift

def run_analysis():

    print("\n=== Stage A: Exploratory Analysis ===")

    corrupt_folder = "../outputs/corrupted"
    noisy_images = sorted(os.listdir(corrupt_folder))

    stats = []

    def analyze_image(img, title):
        mean = np.mean(img)
        var = np.var(img)

        plt.figure(figsize=(12,4))

        # Image
        plt.subplot(1,3,1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')

        # Histogram
        plt.subplot(1,3,2)
        plt.hist(img.ravel(), bins=256, color='gray')
        plt.title(f"Histogram\nMean={mean:.2f}, Var={var:.2f}")

        # DFT Magnitude
        F = fftshift(fft2(img))
        magnitude = np.log(1 + np.abs(F))

        plt.subplot(1,3,3)
        plt.imshow(magnitude, cmap='gray')
        plt.title("DFT Magnitude Spectrum")
        plt.axis('off')

        plt.show()

        return mean, var

    for i, filename in enumerate(noisy_images, start=1):
        path = os.path.join(corrupt_folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        print(f"Executing: Analyzing noisy image {i}")
        mean, var = analyze_image(img, f"Noisy Image {i}")
        stats.append((mean, var))

    print("Success: Stage A analysis completed.")

    return stats
