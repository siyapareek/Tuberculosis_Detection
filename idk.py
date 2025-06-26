import h5py
import numpy as np
import matplotlib.pyplot as plt

# Function to load features from .h5 file
def load_features_from_h5(file_path, dataset_name):
    with h5py.File(file_path, 'r') as h5file:
        dataset = h5file[dataset_name].name
    return dataset

# Replace 'your_file_path.h5' with the actual path to your .h5 file
file_path = "C:/Users/ansh/OneDrive/Desktop/Research/6. SVM/tb.h5"

# Replace 'tuberculosis' and 'non_tuberculosis' with the actual dataset names in the .h5 file
try:
    tuberculosis_features = load_features_from_h5(file_path, 'model_weights')
    non_tuberculosis_features = load_features_from_h5(file_path, 'optimizer_weights')
except ValueError as e:
    print(e)
    exit(1)

# Assuming each feature contains two dimensions for the scatter plot
# If you have more dimensions, you may need to use dimensionality reduction techniques like PCA or t-SNE
# to visualize them in 2D.

# Scatter plot for tuberculosis features
plt.scatter(tuberculosis_features.to_numpy()[:, 0], tuberculosis_features.to_numpy()[:, 1], c='red', label='Tuberculosis')

# Scatter plot for non-tuberculosis features
plt.scatter(non_tuberculosis_features.to_numpy()[:, 0], non_tuberculosis_features.to_numpy()[:, 1], c='blue', label='Non-Tuberculosis')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Visualization of Tuberculosis and Non-Tuberculosis Features')
plt.legend()
plt.show()
