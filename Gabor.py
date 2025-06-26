import cv2
import numpy as np
import os
from skimage.util import img_as_float

def create_gabor_filter(scale, orientation, wavelength, sigma, aspect_ratio=1.0):
    theta = orientation * np.pi / 5.0
    freq = 1.0 / wavelength

    # Calculate the kernel size based on the given sigma
    kernel_size = int(6 * sigma)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create the Gabor kernel
    g_kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, freq, aspect_ratio)

    # Normalize the kernel to have a sum of 1
    g_kernel /= np.sum(g_kernel)

    return g_kernel

def create_gabor_filter_bank(num_scales=8, num_orientations=5, wavelength=10, sigma=5.0, aspect_ratio=1.0):
    gabor_filter_bank = []
    for scale in range(num_scales):
        for orientation in range(num_orientations):
            gabor_filter = create_gabor_filter(scale, orientation, wavelength, sigma, aspect_ratio)
            gabor_filter_bank.append(gabor_filter)

    return gabor_filter_bank

def apply_gabor_filters(image, gabor_filter_bank, save_folder):
    filtered_images = []
    for i, gabor_filter in enumerate(gabor_filter_bank):
        filtered_image = cv2.filter2D(image, cv2.CV_64F, gabor_filter)
        filtered_image /= np.max(filtered_image)  # Normalize to the range [0, 1]
        filtered_image = (filtered_image * 255).astype(np.uint8)  # Scale to [0, 255]
        filtered_images.append(filtered_image)

        # Save the filtered image to the specified folder
        save_path = os.path.join(save_folder, f"Gabor_Filtered_Image_{i}.png")
        cv2.imwrite(save_path, filtered_image)

    return filtered_images

def compute_feature_vector(filtered_images):
    feature_vector = []
    for filtered_image in filtered_images:
        feature_vector.append(np.mean(filtered_image))
        feature_vector.append(np.std(filtered_image))

    return feature_vector

def process_images(image_folder, num_scales=8, num_orientations=5, wavelength=10, sigma=5.0, aspect_ratio=1.0):
    image_paths = os.listdir(image_folder)
    num_images = len(image_paths)

    # Create the Gabor filter bank
    gabor_filter_bank = create_gabor_filter_bank(num_scales, num_orientations, wavelength, sigma, aspect_ratio)

    all_feature_vectors = []
    for image_path in image_paths:
        # Load an image
        image = cv2.imread(os.path.join(image_folder, image_path), cv2.IMREAD_GRAYSCALE)
        image = img_as_float(image)  # Convert image to float in the range [0, 1]

        # Create a folder to save the filtered images
        save_folder = os.path.join("Gabor_Filtered_Images", image_path[:-4])
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Apply Gabor filters to the image and save filtered images
        filtered_images = apply_gabor_filters(image, gabor_filter_bank, save_folder)

        # Compute the feature vector for the image
        feature_vector = compute_feature_vector(filtered_images)

        all_feature_vectors.append(feature_vector)

    return np.vstack(all_feature_vectors)

if __name__ == "__main__":
    # Folder containing 40 images (update this path accordingly)
    image_folder = "C:/Users/ansh/OneDrive/Desktop/Research/4. Gabor Filter Loop/Decomposed Images"

    # Parameters for the Gabor filter bank
    num_scales = 8
    num_orientations = 5
    wavelength = 10
    sigma = 5.0
    aspect_ratio = 1.0

    # Process images and get the feature matrix
    feature_matrix = process_images(image_folder, num_scales, num_orientations, wavelength, sigma, aspect_ratio)

    # Check the shape of the feature matrix
    print("Feature matrix shape:", feature_matrix.shape)

    # Now, you have a feature matrix with shape (40, 1600), where each row represents the feature vector for one image.
    # You can use this feature matrix for further analysis or machine learning tasks.
