# integration.py
import os
import numpy as np
import cv2
import csv

def run_integration(S2, S3):

    print("\n=== Stage G: Final Integration & Robustness ===")

    base_folder = "../dataset/base"
    log_folder = "../logs"
    os.makedirs(log_folder, exist_ok=True)

    base_images = sorted(
        [f for f in os.listdir(base_folder) if f.startswith("base_")]
    )

    perturb_factors = [0.8, 1.2]
    results = []

    for factor in perturb_factors:

        S2_perturbed = int(S2 * factor)

        print(f"Executing: Testing perturbation factor: {factor}")

        for i, filename in enumerate(base_images, start=1):

            img = cv2.imread(
                os.path.join(base_folder, filename),
                cv2.IMREAD_GRAYSCALE
            )

            rows, cols = img.shape
            x = np.linspace(0, 1, cols)

            f = (i + S3) / 50
            phi = i * np.pi / 6

            sigma = S2_perturbed * (
                0.5 + 0.5 * np.sin(2 * np.pi * f * x + phi)
            )

            noise = np.random.randn(rows, cols) * sigma
            noisy = np.clip(img + noise, 0, 255).astype(np.uint8)

            var = np.var(noisy)

            results.append([factor, i, var])

    # ------------------------------
    # Save experimental log (CSV)
    # ------------------------------
    csv_path = os.path.join(log_folder, "experiments.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["factor", "image_index", "variance"])
        writer.writerows(results)

    # Print summary
    for r in results:
        print(f"Factor {r[0]} | Image {r[1]} | Variance {r[2]:.2f}")

    print("Success: Stage G completed.")
