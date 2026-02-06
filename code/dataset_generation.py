# dataset_generation.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def run_dataset():
    print("\n=== Stage 0–2: Dataset Generation ===")

    # ------------------------------
    # Stage 0: Student Seed & Parameters
    # ------------------------------
    reg_no = "P23EGCP8034"

    digits = [int(d) for d in reg_no if d.isdigit()]
    seed = int("".join(map(str, digits[:6])))

    S1 = (seed % 7) + 3
    S2 = ((seed // 7) % 13) + 1
    S3 = ((seed // 91) % 11) + 2

    print(f"Student seed: {seed}")
    print(f"S1 = {S1}")
    print(f"S2 = {S2}")
    print(f"S3 = {S3}")

    # Save parameters globally (for other modules)
    globals().update({"seed": seed, "S1": S1, "S2": S2, "S3": S3})

    # ------------------------------
    # Stage 1: Base Image Loading
    # ------------------------------
    base_folder = "../dataset/base"
    os.makedirs(base_folder, exist_ok=True)

    valid_ext = (".png", ".jpg", ".jpeg", ".bmp")
    base_images = sorted([
        f for f in os.listdir(base_folder)
        if f.lower().endswith(valid_ext) and not f.startswith("base_")
    ])

    assert len(base_images) >= 6, "Error: Put at least 6 original images in images/base/"

    print("Success: Loading student base images...")

    for i, filename in enumerate(base_images[:6]):
        path = os.path.join(base_folder, filename)

        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Error: Cannot read image: {filename}")

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, (512, 512))

        save_path = os.path.join(base_folder, f"base_{i + 1}.png")
        cv2.imwrite(save_path, img)

        plt.figure(figsize=(4, 4))
        plt.imshow(img, cmap="gray")
        plt.title(f"Base Image {i + 1}")
        plt.axis("off")
        plt.show()

    print("Success: Stage 1 complete: Base images prepared.")

    # ------------------------------
    # Stage 2: Spatially Varying Gaussian Noise
    # ------------------------------
    corrupt_folder = "../outputs/corrupted"
    os.makedirs(corrupt_folder, exist_ok=True)

    base_images = sorted([f for f in os.listdir(base_folder) if f.startswith("base_")])

    def spatially_varying_gaussian_noise(image, i, S2, S3):
        rows, cols = image.shape
        x = np.linspace(0, 1, cols)

        f = (i + S3) / 50
        phi = i * np.pi / 6

        sigma = S2 * (0.5 + 0.5 * np.sin(2 * np.pi * f * x + phi))
        noise = np.random.randn(rows, cols) * sigma

        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    print("Executing: Applying corruption pipeline (noise → blur → JPEG)...")

    def motion_blur(img, length, angle):
        psf = np.zeros((length, length))

        center = length // 2
        for i in range(length):
            psf[center, i] = 1

        # rotate PSF
        M = cv2.getRotationMatrix2D((center, center), angle, 1)
        psf = cv2.warpAffine(psf, M, (length, length))

        psf = psf / psf.sum()

        blurred = cv2.filter2D(img, -1, psf)
        return blurred

    for i, filename in enumerate(base_images, start=1):
        path = os.path.join(base_folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # 1 Spatial Gaussian noise
        noisy_img = spatially_varying_gaussian_noise(img, i, S2, S3)

        # 2 Motion blur
        L = S1 + (i % 3)
        theta = (seed % 180) + 7 * i
        blurred_img = motion_blur(noisy_img, L, theta)

        # 3 JPEG compression (supervisor formula)
        Q = 40 + 5 * S3 + (3 * i % 10)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), Q]

        temp_path = os.path.join(corrupt_folder, f"temp_{i}.jpg")
        cv2.imwrite(temp_path, blurred_img, encode_param)

        compressed = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
        os.remove(temp_path)

        # 4 Save final corrupted image
        save_path = os.path.join(corrupt_folder, f"noisy_{i}.png")
        cv2.imwrite(save_path, compressed)

    print("Success: Stage 2 complete: Noisy images saved.")
    return seed, S1, S2, S3