import numpy as np
import pickle
from scipy.optimize import minimize
from math import sqrt
import time
import matplotlib.pyplot as plt


# Initialize Weights Function
def initializeWeights(n_in, n_out):
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


# Sigmoid Activation Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Data Preprocessing Function with Removal of Constant Features
def preprocess():
    # Load dataset
    pickle_obj = pickle.load(open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']

    # Remove features with zero variance
    feature_variance = np.var(features, axis=0)
    retained_indices = np.where(feature_variance > 0)[0]
    features = features[:, retained_indices]

    # Save retained indices to file
    np.savetxt('face_all_retained_indices.txt', retained_indices, fmt='%d')

    # Split dataset into training, validation, and test sets
    train_x = features[0:21100] / 255.0
    valid_x = features[21100:23765] / 255.0
    test_x = features[23765:] / 255.0

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]

    return train_x, train_y, valid_x, valid_y, test_x, test_y, retained_indices


# Objective Function for Training the Neural Network
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    n = training_data.shape[0]

    # Forward Propagation
    bias = np.ones((n, 1))
    training_data = np.hstack((training_data, bias))
    z = sigmoid(np.dot(training_data, w1.T))
    z = np.hstack((z, np.ones((z.shape[0], 1))))
    o = sigmoid(np.dot(z, w2.T))

    # One-hot encode the training labels
    training_label_one_hot = np.zeros((training_label.shape[0], n_class))
    training_label_one_hot[np.arange(training_label.shape[0]), training_label.astype(int)] = 1

    # Compute Error Function with Regularization
    o = np.clip(o, 1e-10, 1 - 1e-10)
    error = -np.sum(training_label_one_hot * np.log(o) + (1 - training_label_one_hot) * np.log(1 - o)) / n
    regularization = (lambdaval / (2 * n)) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    obj_val = error + regularization

    # Backpropagation
    delta = o - training_label_one_hot
    grad_w2 = np.dot(delta.T, z) / n + (lambdaval / n) * w2

    grad_hidden = np.dot(delta, w2[:, :-1]) * z[:, :-1] * (1 - z[:, :-1])
    grad_w1 = np.dot(grad_hidden.T, training_data) / n + (lambdaval / n) * w1

    # Flatten gradients to a single vector
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)

    return obj_val, obj_grad


# Function to Predict Labels using Trained Weights
def nnPredict(w1, w2, data):
    n = data.shape[0]
    bias = np.ones((n, 1))
    data = np.hstack((data, bias))

    # Forward Propagation
    z = sigmoid(np.dot(data, w1.T))
    z = np.hstack((z, np.ones((z.shape[0], 1))))
    o = sigmoid(np.dot(z, w2.T))

    # Predict labels
    labels = np.argmax(o, axis=1)
    return labels


"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label, selected_features = preprocess()

# Hyperparameter Tuning
hidden_units_list = [50, 100, 200, 400, 800]  # Different values for hidden units
lambdaval_list = list(range(0, 105, 5))  # Different values for lambda (incremented by 5)

best_validation_accuracy = 0
best_hyperparameters = ()

results = []

for n_hidden in hidden_units_list:
    for lambdaval in lambdaval_list:
        print(f"\nTraining with {n_hidden} hidden units and λ = {lambdaval}...")

        # Initialize weights
        initial_w1 = initializeWeights(train_data.shape[1], n_hidden)
        initial_w2 = initializeWeights(n_hidden, 2)

        # Unroll weights into a single vector
        initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

        # Set optimization options and train the model
        args = (train_data.shape[1], n_hidden, 2, train_data, train_label, lambdaval)
        opts = {'maxiter': 50}
        start_time = time.time()
        nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
        training_time = time.time() - start_time

        # Reshape trained weights back into w1 and w2 matrices
        params = nn_params.x
        w1 = params[0:n_hidden * (train_data.shape[1] + 1)].reshape((n_hidden, (train_data.shape[1] + 1)))
        w2 = params[(n_hidden * (train_data.shape[1] + 1)):].reshape((2, (n_hidden + 1)))

        # Predict labels on validation and test data
        predicted_label_validation = nnPredict(w1, w2, validation_data)
        validation_accuracy = 100 * np.mean((predicted_label_validation == validation_label).astype(float))

        predicted_label_test = nnPredict(w1, w2, test_data)
        test_accuracy = 100 * np.mean((predicted_label_test == test_label).astype(float))

        print(f"Validation set Accuracy: {validation_accuracy}%")
        print(f"Test set Accuracy: {test_accuracy}%")

        results.append({
            'n_hidden': n_hidden,
            'lambdaval': lambdaval,
            'validation_accuracy': validation_accuracy,
            'test_accuracy': test_accuracy,
            'training_time': training_time
        })

        # Save the best model
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_hyperparameters = (n_hidden, lambdaval, w1, w2)

# Save the best model parameters
n_hidden, lambdaval, best_w1, best_w2 = best_hyperparameters
params = {
    'selected_features': selected_features,
    'optimal_n_hidden': n_hidden,
    'w1': best_w1,
    'w2': best_w2,
    'optimal_lambda': lambdaval
}

with open('params_face_all.pickle', 'wb') as f:
    pickle.dump(params, f)

# Save all training details in one pickle file
with open('training_details_face_all.pickle', 'wb') as f:
    pickle.dump({'results': results}, f)

print("\nBest Model Saved to params_face_all.pickle")
print("Training Details Saved to training_details_face_all.pickle")

# Plotting Results
# Regularization λ vs. Validation Accuracy
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
for n_hidden in hidden_units_list:
    lambdas = [res['lambdaval'] for res in results if res['n_hidden'] == n_hidden]
    accuracies = [res['validation_accuracy'] for res in results if res['n_hidden'] == n_hidden]
    plt.plot(lambdas, accuracies, marker='o', label=f'{n_hidden} Hidden Units')
plt.xlabel('Regularization λ')
plt.ylabel('Validation Accuracy (%)')
plt.title('Regularization λ vs. Validation Accuracy')
plt.legend()

# Number of Hidden Units vs. Training Time
plt.subplot(1, 2, 2)
for lambdaval in lambdaval_list:
    hidden_units = [res['n_hidden'] for res in results if res['lambdaval'] == lambdaval]
    times = [res['training_time'] for res in results if res['lambdaval'] == lambdaval]
    plt.plot(hidden_units, times, marker='o', label=f'λ = {lambdaval}')
plt.xlabel('Number of Hidden Units')
plt.ylabel('Training Time (s)')
plt.title('Number of Hidden Units vs. Training Time')
plt.legend()
plt.tight_layout()
plt.savefig('face_all_hidden_units_vs_training_time.png')
print("Plots saved as 'face_all_hidden_units_vs_training_time.png'")

# Plot Validation Accuracy vs. Test Accuracy Comparison
plt.figure(figsize=(15, 10))
for n_hidden in hidden_units_list:
    validation_accuracies = [res['validation_accuracy'] for res in results if res['n_hidden'] == n_hidden]
    test_accuracies = [res['test_accuracy'] for res in results if res['n_hidden'] == n_hidden]
    lambdas = [res['lambdaval'] for res in results if res['n_hidden'] == n_hidden]

    plt.plot(lambdas, validation_accuracies, linestyle='-', marker='o',
             label=f'Validation Accuracy (Hidden Units: {n_hidden})', alpha=0.7)
    plt.plot(lambdas, test_accuracies, linestyle='--', marker='o', label=f'Test Accuracy (Hidden Units: {n_hidden})',
             alpha=0.7)

plt.xlabel('Regularization λ')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy vs. Test Accuracy for Different Hidden Units')
plt.legend(loc='lower right')
plt.grid(visible=True)
plt.savefig('validation_vs_test_accuracy_comparison.png')
plt.show()
print("Plot saved as 'validation_vs_test_accuracy_comparison.png'")
