# Tuberculosis Detection Using Chest X-Rays

## Project Overview
Globally, tuberculosis (TB) continues to be a serious public health concern, especially in areas with poor access to diagnostic resources. The goal of this project is to create a machine learning-based method for accurately detecting tuberculosis from chest X-rays. Utilizing deep learning models and sophisticated image processing techniques, we achieved a proof-of-concept system with an astounding **96.1% accuracy rate**.

## Goal
The primary objective is to develop an automated system capable of differentiating between normal chest X-rays and those showing signs of tuberculosis. By ensuring high resilience and efficiency in the detection process, the project aims to address key challenges in medical imaging.

## Data Collection
The dataset used for this project was sourced from Kaggle and comprised over **1,000 chest X-ray images**, including both normal and tuberculosis-affected cases. This well-balanced dataset provided a solid foundation for training and evaluating the model.

## Data Preprocessing
A comprehensive preprocessing pipeline was implemented to ensure the quality and usability of the dataset:

1. **Noise Reduction**:
   - Chest X-ray images often contain noise, which can hinder accurate feature extraction. Advanced noise reduction techniques were employed to improve image clarity while preserving essential details.

2. **Feature Extraction with Compact Variational Mode Decomposition (VMD)**:
   - Images were decomposed into intrinsic mode functions (IMFs) using compact VMD. This step helped identify critical patterns relevant to TB detection.

3. **Gabor Filter Bank**:
   - Texture features were extracted using a Gabor filter bank. The images were oriented into an 8x5 grid of orientations to capture diverse textural characteristics vital for distinguishing TB-affected lungs. Final features were stored in `.h5` files for efficient storage and processing.

## Model Development

1. **Deep Learning Model Architecture**:
   - A deep learning model with a **nine-layer architecture** was developed. This design balanced computational efficiency and complexity, featuring convolutional layers for feature extraction, pooling layers for dimensionality reduction, and fully connected layers for classification.

2. **Training and Evaluation**:
   - The model was trained on a stratified dataset, ensuring equal representation of TB and normal cases.
   - The **Adam optimizer** and **cross-entropy loss** were used to achieve effective convergence.

3. **Accuracy Achievement**:
   - The model achieved an impressive **96.1% accuracy**, demonstrating its efficacy in detecting tuberculosis from chest X-rays.

## Classification and Validation
To further validate the model, three additional classifiers were integrated:

1. **Support Vector Machine (SVM)**:
   - Provided robust decision boundaries for highly accurate TB detection.

2. **Random Forest**:
   - Improved accuracy and stability through ensemble-based predictions from multiple decision trees.

3. **Decision Tree**:
   - Allowed for early-stage validation and analysis of feature extraction capabilities.

## Visualization
Scatter plots were used to visualize classification results, highlighting the distribution of normal and TB cases. These plots demonstrated the effectiveness of the feature extraction and classification pipeline, with clusters corresponding to different categories.

## Achievements
- Developed a proof-of-concept model with an accuracy of **96.1%**.
- Implemented a robust data preprocessing pipeline to enhance data quality.
- Validated the model using multiple classifiers to ensure reliability.
- Demonstrated the potential of machine learning in medical imaging applications.

## Future Work

1. **Dataset Expansion**:
   - Incorporate a larger and more diverse dataset to improve generalizability.

2. **Model Optimization**:
   - Explore advanced architectures like transformers and perform hyperparameter tuning to achieve further accuracy improvements.

3. **Deployment**:
   - Develop a user-friendly interface for radiologists to quickly screen for tuberculosis in real-world clinical settings.

4. **Explainability**:
   - Integrate explainable AI techniques to provide insights into model decisions, fostering trust among medical professionals.

---
This project showcases the power of machine learning in addressing critical challenges in healthcare, paving the way for accessible and reliable TB detection solutions.
