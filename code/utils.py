import os
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift

def ensure_folder(path):
    os.makedirs(path, exist_ok=True)

def load_gray(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def show_image(img, title="Image"):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def image_stats(img):
    return np.mean(img), np.var(img)

def dft_magnitude(img):
    F = fftshift(fft2(img))
    return np.log(1 + np.abs(F))

def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
