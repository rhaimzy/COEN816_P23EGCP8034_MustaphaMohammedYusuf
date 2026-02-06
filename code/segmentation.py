# ------------------------------
# Stage F: Segmentation and Morphology
# ------------------------------

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt# segmentation.py
import os
import cv2
import numpy as np

def run_segmentation():

    print("\n=== Stage F: Segmentation and Morphology ===")

    final_folder = "../outputs/final"
    os.makedirs(final_folder, exist_ok=True)

    restored_images = sorted(
        [f for f in os.listdir(final_folder) if f.startswith("wiener_")]
    )

    # ------------------------------
    # Edge Detection
    # ------------------------------
    def edge_detection(img):
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        return np.clip(mag, 0, 255).astype(np.uint8)

    # ------------------------------
    # Manual Otsu Threshold
    # ------------------------------
    def global_threshold(img):
        hist = np.bincount(img.ravel(), minlength=256)
        total = img.size
        sum_total = np.dot(np.arange(256), hist)

        sumB = 0
        wB = 0
        max_var = 0
        threshold = 0

        for t in range(256):
            wB += hist[t]
            if wB == 0:
                continue
            wF = total - wB
            if wF == 0:
                break
            sumB += t * hist[t]
            mB = sumB / wB
            mF = (sum_total - sumB) / wF
            var_between = wB * wF * (mB - mF)**2

            if var_between > max_var:
                max_var = var_between
                threshold = t

        binary = np.zeros_like(img)
        binary[img >= threshold] = 255
        return binary

    # ------------------------------
    # Local Adaptive Threshold
    # ------------------------------
    def local_threshold(img, window=15, C=5):
        pad = window // 2
        padded = np.pad(img, pad, mode='reflect')
        result = np.zeros_like(img)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                local = padded[i:i+window, j:j+window]
                result[i, j] = 255 if img[i,j] > np.mean(local) - C else 0

        return result

    # ------------------------------
    # Morphological Refinement
    # ------------------------------
    def morphology_refine(binary):
        kernel = np.ones((3,3), np.uint8)
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        return opening

    print("Executing: Running segmentation...")

    for i, filename in enumerate(restored_images, start=1):
        img = cv2.imread(os.path.join(final_folder, filename), cv2.IMREAD_GRAYSCALE)

        edges = edge_detection(img)
        global_bin = global_threshold(img)
        local_bin = local_threshold(img)
        refined = morphology_refine(local_bin)

        cv2.imwrite(os.path.join(final_folder, f"edges_{i}.png"), edges)
        cv2.imwrite(os.path.join(final_folder, f"global_seg_{i}.png"), global_bin)
        cv2.imwrite(os.path.join(final_folder, f"local_seg_{i}.png"), refined)

    print("Success: Stage F completed.")
