import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import tkinter as tk
from PIL import Image, ImageTk

def compact_vmd(signal, alpha, tau, K):
    # Remaining code for compact_vmd function...
    N = len(signal)
    u = signal
    omega = np.zeros((K, N))
    residue = np.zeros(N)

    for i in range(K):
        u_hat = np.fft.fft(u)
        u_hat_temp = np.fft.fftshift(u_hat)

        omega[i, :] = np.real(np.fft.ifft(u_hat_temp))
        residue = residue + omega[i, :]
        u = u - alpha * residue

        if tau >= 0:
            u = np.where(u > tau, u - tau, 0)
        else:
            u = np.where(u < tau, u - tau, 0)

    return omega, residue

# Define the parameters for compact VMD decomposition
alpha = 200  # Penalization parameter
tau = 0.02   # Threshold for positive sparse residue
K = 5        # Number of modes to decompose

# Function to save the last decomposed image
def save_last_decomposed_image(omega, image_path):
    # Select the last decomposed image
    last_decomposed_image = omega[-1, :].reshape(image.shape)

    # Normalize the decomposed image to [0, 255]
    last_decomposed_image = cv2.normalize(last_decomposed_image, None, 0, 255, cv2.NORM_MINMAX)

    # Create the output folder if it doesn't exist
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    # Extract the image name from the path
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Save the last decomposed image
    output_path = os.path.join(output_dir, f"{image_name}_last_decomposed.png")
    cv2.imwrite(output_path, last_decomposed_image)

# Process all images in the specified folder
input_folder = "C:/Users/ansh/OneDrive/Desktop/Research/Compact VMD/Dataset"
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)

    # Read the image in grayscale and resize if necessary
    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, (512, 512))

    # Perform compact VMD decomposition
    omega, residue = compact_vmd(image.flatten(), alpha, tau, K)

    # Save the last decomposed image
    save_last_decomposed_image(omega, image_path)

# Rest of the code for GUI display remains unchanged
# ...
window = tk.Tk()

# Run the GUI main loop
window.mainloop()
