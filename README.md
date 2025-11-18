#Project Based Experiments
## Objective :
 Build a Multilayer Perceptron (MLP) to classify handwritten digits in python
## Steps to follow:
## Dataset Acquisition:
Download the MNIST dataset. You can use libraries like TensorFlow or PyTorch to easily access the dataset.
## Data Preprocessing:
Normalize pixel values to the range [0, 1].
Flatten the 28x28 images into 1D arrays (784 elements).
## Data Splitting:

Split the dataset into training, validation, and test sets.
Model Architecture:
## Design an MLP architecture. 
You can start with a simple architecture with one input layer, one or more hidden layers, and an output layer.
Experiment with different activation functions, such as ReLU for hidden layers and softmax for the output layer.
## Compile the Model:
Choose an appropriate loss function (e.g., categorical crossentropy for multiclass classification).Select an optimizer (e.g., Adam).
Choose evaluation metrics (e.g., accuracy).
## Training:
Train the MLP using the training set.Use the validation set to monitor the model's performance and prevent overfitting.Experiment with different hyperparameters, such as the number of hidden layers, the number of neurons in each layer, learning rate, and batch size.
## Evaluation:

Evaluate the model on the test set to get a final measure of its performance.Analyze metrics like accuracy, precision, recall, and confusion matrix.
## Fine-tuning:
If the model is not performing well, experiment with different architectures, regularization techniques, or optimization algorithms to improve performance.
## Visualization:
Visualize the training/validation loss and accuracy over epochs to understand the training process. Visualize some misclassified examples to gain insights into potential improvements.

# Program:
```python

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data.astype(np.float32)
y = mnist.target.astype(int)

X = X / 255.0

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    batch_size=256,
    learning_rate_init=1e-3,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
    tol=1e-4,
    verbose=True,
    random_state=42
)

mlp.fit(X_train_s, y_train)

y_val_pred = mlp.predict(X_val_s)
y_test_pred = mlp.predict(X_test_s)
print("Val accuracy:", accuracy_score(y_val, y_val_pred))
print("Test accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification report (test):\n", classification_report(y_test, y_test_pred))

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(9,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (test)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

plt.figure()
plt.plot(mlp.loss_curve_)
plt.title("Training Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

mis_idx = np.where(y_test_pred != y_test)[0]
print("Total misclassified (test):", len(mis_idx))
n_show = min(12, len(mis_idx))
plt.figure(figsize=(12, 6))
for i, idx in enumerate(mis_idx[:n_show]):
    ax = plt.subplot(3, 4, i+1)
    ax.imshow((X_test[idx].reshape(28,28)), cmap='gray')
    ax.set_title(f"True: {y_test[idx]}, Pred: {y_test_pred[idx]}")
    ax.axis('off')
plt.tight_layout()
plt.show()

correct_idx = np.where(y_test_pred == y_test)[0]
print("Total correctly classified:", len(correct_idx))
n_show = min(12, len(correct_idx))
plt.figure(figsize=(12, 6))

for i, idx in enumerate(correct_idx[:n_show]):
    ax = plt.subplot(3, 4, i+1)
    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    ax.set_title(f"True & Pred: {y_test[idx]}")
    ax.axis('off')

plt.tight_layout()
plt.show()
```

## Output:
### Training

<img width="892" height="711" alt="image" src="https://github.com/user-attachments/assets/c4ec4710-56e2-4bc8-90b8-a78442f26229" />

### Accuracy
<img width="518" height="446" alt="image" src="https://github.com/user-attachments/assets/f91bf344-8c82-4944-86c1-2da5e6b7e77b" />

### Confusion Matrix
<img width="890" height="735" alt="image" src="https://github.com/user-attachments/assets/4096b093-5f41-4fca-b146-1c2a9c8efa30" />


### Loss
<img width="745" height="545" alt="image" src="https://github.com/user-attachments/assets/8a1ef6a5-2d24-4e03-a66f-69dc6ad4fa13" />
### Classification
<img width="950" height="601" alt="image" src="https://github.com/user-attachments/assets/972965ac-3c8f-4646-a497-3b0ad2dddbfb" />
### Misclassification

<img width="982" height="581" alt="image" src="https://github.com/user-attachments/assets/72193fbc-1725-4922-b7b5-46378cdbe675" />
