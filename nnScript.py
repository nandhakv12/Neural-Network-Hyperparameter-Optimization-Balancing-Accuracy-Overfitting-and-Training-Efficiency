import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
import pickle
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
def preprocess(validation_size=10000):
    mat = loadmat('mnist_all.mat')

    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(validation_size, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(validation_size,))
    test_label_preprocess = np.zeros(shape=(10000,))

    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0

    # Split dataset into training, validation, and test sets
    for key in mat:
        if "train" in key:
            label = int(key[-1])
            tup = mat.get(key)
            tup_perm = np.random.permutation(range(tup.shape[0]))
            tup_len = len(tup)
            tag_len = tup_len - validation_size // 10

            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[validation_size // 10:], :]
            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_len += tag_len
            train_label_len += tag_len

            validation_preprocess[validation_len:validation_len + validation_size // 10] = tup[tup_perm[:validation_size // 10], :]
            validation_label_preprocess[validation_label_len:validation_label_len + validation_size // 10] = label
            validation_len += validation_size // 10
            validation_label_len += validation_size // 10

        elif "test" in key:
            label = int(key[-1])
            tup = mat.get(key)
            tup_perm = np.random.permutation(range(tup.shape[0]))
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len

    combined_data = np.vstack((train_preprocess, validation_preprocess, test_preprocess))
    non_constant_indices = np.var(combined_data, axis=0) > 0

    train_data = train_preprocess[:, non_constant_indices] / 255.0
    validation_data = validation_preprocess[:, non_constant_indices] / 255.0
    test_data = test_preprocess[:, non_constant_indices] / 255.0

    train_label = train_label_preprocess
    validation_label = validation_label_preprocess
    test_label = test_label_preprocess

    train_label_one_hot = np.zeros((train_label.shape[0], 10))
    train_label_one_hot[np.arange(train_label.shape[0]), train_label.astype(int)] = 1

    validation_label_one_hot = np.zeros((validation_label.shape[0], 10))
    validation_label_one_hot[np.arange(validation_label.shape[0]), validation_label.astype(int)] = 1

    test_label_one_hot = np.zeros((test_label.shape[0], 10))
    test_label_one_hot[np.arange(test_label.shape[0]), test_label.astype(int)] = 1

    return train_data, train_label_one_hot, validation_data, validation_label_one_hot, test_data, test_label_one_hot, non_constant_indices

# Objective Function for Training the Neural Network
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    n = training_data.shape[0]

    bias = np.ones((n, 1))
    training_data = np.hstack((training_data, bias))
    z = sigmoid(np.dot(training_data, w1.T))
    z = np.hstack((z, np.ones((z.shape[0], 1))))
    o = sigmoid(np.dot(z, w2.T))

    o = np.clip(o, 1e-10, 1 - 1e-10)

    error = -np.sum(training_label * np.log(o) + (1 - training_label) * np.log(1 - o)) / n
    regularization = (lambdaval / (2 * n)) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    obj_val = error + regularization

    delta = o - training_label
    grad_w2 = np.dot(delta.T, z) / n + (lambdaval / n) * w2
    grad_hidden = np.dot(delta, w2[:, :-1]) * z[:, :-1] * (1 - z[:, :-1])
    grad_w1 = np.dot(grad_hidden.T, training_data) / n + (lambdaval / n) * w1

    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)

    return obj_val, obj_grad

# Function to Predict Labels using Trained Weights
def nnPredict(w1, w2, data):
    n = data.shape[0]
    bias = np.ones((n, 1))
    data = np.hstack((data, bias))
    z = sigmoid(np.dot(data, w1.T))
    z = np.hstack((z, np.ones((z.shape[0], 1))))
    o = sigmoid(np.dot(z, w2.T))
    labels = np.argmax(o, axis=1)
    return labels

# Main Experimentation Code
train_data, train_label, validation_data, validation_label, test_data, test_label, selected_features = preprocess()

n_input = train_data.shape[1]
n_class = 10
hidden_units_list = [4, 8, 12, 16, 20]
lambdaval_list = list(range(0, 65, 5))

best_validation_accuracy = 0
best_model = None
results = []
step_details_all = []  # List to store details of each step for analysis later

# Loop through each combination of hyperparameters
for n_hidden in hidden_units_list:
    for lambdaval in lambdaval_list:
        print(f"\nTraining with {n_hidden} hidden units and λ = {lambdaval}...")
        initial_w1 = initializeWeights(n_input, n_hidden)
        initial_w2 = initializeWeights(n_hidden, n_class)
        initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

        args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)
        opts = {'maxiter': 50}
        start_time = time.time()
        nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
        training_time = time.time() - start_time

        w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
        w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

        predicted_label_train = nnPredict(w1, w2, train_data)
        train_accuracy = 100 * np.mean((predicted_label_train == np.argmax(train_label, axis=1)).astype(float))

        predicted_label_validation = nnPredict(w1, w2, validation_data)
        validation_accuracy = 100 * np.mean((predicted_label_validation == np.argmax(validation_label, axis=1)).astype(float))

        predicted_label_test = nnPredict(w1, w2, test_data)
        test_accuracy = 100 * np.mean((predicted_label_test == np.argmax(test_label, axis=1)).astype(float))

        # Store results in list
        step_details_all.append({
            'n_hidden': n_hidden,
            'lambdaval': lambdaval,
            'train_accuracy': train_accuracy,
            'validation_accuracy': validation_accuracy,
            'test_accuracy': test_accuracy,
            'training_time': training_time
        })

        print(f"Train Accuracy: {train_accuracy}%")
        print(f"Validation Accuracy: {validation_accuracy}%")
        print(f"Test Accuracy: {test_accuracy}%")
        print(f"Training Time: {training_time:.2f} seconds")

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_model = (w1, w2, n_hidden, lambdaval)

# Save the best model parameters in params.pickle
w1, w2, optimal_n_hidden, optimal_lambda = best_model
params = {
    'selected_features': selected_features,
    'optimal_n_hidden': optimal_n_hidden,
    'w1': w1,
    'w2': w2,
    'optimal_lambda': optimal_lambda
}

with open('params.pickle', 'wb') as f:
    pickle.dump(params, f)

# Save all step details for future analysis in step_details_all.pickle
with open('step_details_all.pickle', 'wb') as f:
    pickle.dump(step_details_all, f)

print("\nBest Model Saved to params.pickle")
print("All step details saved to step_details_all.pickle")

# Plotting Results
# Regularization λ vs. Train, Validation, and Test Accuracy
plt.figure(figsize=(15, 5))

# Separate plots for better visualization
for metric, ylabel, title, filename in zip(
        ['train_accuracy', 'validation_accuracy', 'test_accuracy'],
        ['Training Accuracy (%)', 'Validation Accuracy (%)', 'Test Accuracy (%)'],
        ['Regularization λ vs. Training Accuracy', 'Regularization λ vs. Validation Accuracy', 'Regularization λ vs. Test Accuracy'],
        ['lambda_vs_training_accuracy.png', 'lambda_vs_validation_accuracy.png', 'lambda_vs_test_accuracy.png']):

    plt.figure(figsize=(10, 6))
    for n_hidden in hidden_units_list:
        lambdas = [res['lambdaval'] for res in step_details_all if res['n_hidden'] == n_hidden]
        accuracies = [res[metric] for res in step_details_all if res['n_hidden'] == n_hidden]
        plt.plot(lambdas, accuracies, marker='o', label=f'{n_hidden} Hidden Units')
    plt.xlabel('Regularization λ')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    print(f"Plot saved as '{filename}'")

# Number of Hidden Units vs. Training Time
plt.figure(figsize=(10, 6))
for lambdaval in lambdaval_list:
    hidden_units = [res['n_hidden'] for res in step_details_all if res['lambdaval'] == lambdaval]
    times = [res['training_time'] for res in step_details_all if res['lambdaval'] == lambdaval]
    plt.plot(hidden_units, times, marker='o', label=f'λ = {lambdaval}')
plt.xlabel('Number of Hidden Units')
plt.ylabel('Training Time (s)')
plt.title('Number of Hidden Units vs. Training Time')
plt.legend()
plt.savefig('hidden_units_vs_training_time.png')

print("Plots for accuracy and training time saved.")