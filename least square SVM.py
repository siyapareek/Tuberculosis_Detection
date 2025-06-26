import h5py
import matplotlib.pyplot as plt

def visualize_h5_data(h5_file_path, dataset_name):
    # Open the .h5 file in read mode
    with h5py.File(h5_file_path, 'r') as h5_file:
        # Check if the dataset exists in the file
        if dataset_name not in h5_file:
            print(f"Dataset '{dataset_name}' not found in the .h5 file.")
            return

        # Access the dataset
        dataset = h5_file[dataset_name]

        # Read the data from the dataset into a NumPy array
        data = dataset[:]

        # Check the shape of the data (if necessary)
        print("Data shape:", data.shape)

        # Visualization
        plt.figure()
        plt.plot(data)
        plt.xlabel('Sample Index')
        plt.ylabel('Feature Value')
        plt.title(f'Feature Extraction Data: {dataset_name}')
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    # Provide the path to your .h5 file and the dataset name you want to visualize
    h5_file_path = "C:/Users/ansh/OneDrive/Desktop/Research/6. SVM/tb.h5"
    dataset_name = 'tuberculosis'

    visualize_h5_data(h5_file_path, dataset_name)
