import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Load the extracted features from the files
train_features = np.load('train_features.npy')
test_features = np.load('test_features.npy')

# Reshape the features to match the input format expected by the decision tree classifier
train_features = train_features.reshape(train_features.shape[0], -1)
test_features = test_features.reshape(test_features.shape[0], -1)

# Synthetic labels for cross-validation (replace with actual labels if available)
# Assuming the number of training samples is train_features.shape[0]
synthetic_labels = np.random.choice(['tuberculosis', 'non-tuberculosis'], size=train_features.shape[0])

# Create a Decision Tree classifier
decision_tree = DecisionTreeClassifier()

# Perform cross-validation to estimate the classifier's performance
# Here, 'cv' is the number of cross-validation folds, you can adjust it as needed
cv_scores = cross_val_score(decision_tree, train_features, synthetic_labels, cv=5)

# Calculate the mean accuracy across all folds
mean_accuracy = np.mean(cv_scores)

print("Cross-Validation Mean Accuracy:", mean_accuracy)
