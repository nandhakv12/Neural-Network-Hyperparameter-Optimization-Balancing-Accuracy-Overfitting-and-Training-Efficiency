import tensorflow as tf
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt


# Create model with variable hidden layers
class MultilayerPerceptron(tf.Module):
    def __init__(self, n_layers=3, name=None):
        super().__init__(name=name)
        self.n_layers = n_layers
        self.n_input = 2376  # data input
        self.n_hidden = 256  # hidden layer features
        self.n_classes = 2

        # Initialize weights & biases for each hidden layer and output layer
        self.weights = {}
        self.biases = {}

        # Input layer to first hidden layer
        self.weights['h1'] = tf.Variable(tf.random.normal([self.n_input, self.n_hidden], dtype=tf.float32))
        self.biases['b1'] = tf.Variable(tf.random.normal([self.n_hidden], dtype=tf.float32))

        # Hidden layers
        for i in range(2, n_layers + 1):
            self.weights[f'h{i}'] = tf.Variable(tf.random.normal([self.n_hidden, self.n_hidden], dtype=tf.float32))
            self.biases[f'b{i}'] = tf.Variable(tf.random.normal([self.n_hidden], dtype=tf.float32))

        # Output layer
        self.weights['out'] = tf.Variable(tf.random.normal([self.n_hidden, self.n_classes], dtype=tf.float32))
        self.biases['out'] = tf.Variable(tf.random.normal([self.n_classes], dtype=tf.float32))

    def __call__(self, x):
        # Input layer to first hidden layer with RELU activation
        layer = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        layer = tf.nn.relu(layer)

        # Hidden layers
        for i in range(2, self.n_layers + 1):
            layer = tf.add(tf.matmul(layer, self.weights[f'h{i}']), self.biases[f'b{i}'])
            layer = tf.nn.relu(layer)

        # Output layer with linear activation
        out_layer = tf.matmul(layer, self.weights['out']) + self.biases['out']
        return out_layer


# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels.T
    train_y = np.zeros(shape=(21100, 2))
    train_l = labels[0:21100]
    valid_y = np.zeros(shape=(2665, 2))
    valid_l = labels[21100:23765]
    test_y = np.zeros(shape=(2642, 2))
    test_l = labels[23765:]
    for i in range(train_y.shape[0]):
        train_y[i, train_l[i]] = 1
    for i in range(valid_y.shape[0]):
        valid_y[i, valid_l[i]] = 1
    for i in range(test_y.shape[0]):
        test_y[i, test_l[i]] = 1

    return train_x.astype(np.float32), train_y.astype(np.float32), valid_x.astype(np.float32), valid_y.astype(
        np.float32), test_x.astype(np.float32), test_y.astype(np.float32)


# Parameters
learning_rate = 0.0001
training_epochs = 100
batch_size = 100

# Load data
train_features, train_labels, valid_features, valid_labels, test_features, test_labels = preprocess()

# List to hold training metrics for different layer settings
training_metrics_all_layers = []

# Iterate over different numbers of hidden layers
for n_layers in [1,3,5,7]:
    print(f"\nTraining model with {n_layers} hidden layers...")

    # Construct model
    mlp = MultilayerPerceptron(n_layers=n_layers)

    # Define loss and optimizer
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Initialize metrics storage
    training_metrics = {
        'n_layers': n_layers,
        'epoch_losses': [],
        'validation_accuracies': [],
        'epoch_times': [],
        'test_accuracy': None
    }

    # Training loop
    for epoch in range(training_epochs):
        start_time = time.time()
        avg_cost = 0.0
        total_batch = int(train_features.shape[0] / batch_size)

        # Loop over all batches
        for i in range(total_batch):
            batch_x = train_features[i * batch_size: (i + 1) * batch_size]
            batch_y = train_labels[i * batch_size: (i + 1) * batch_size]

            # Run optimization and compute average loss
            with tf.GradientTape() as tape:
                predictions = mlp(batch_x)
                loss = loss_object(batch_y, predictions)
            gradients = tape.gradient(loss, mlp.trainable_variables)
            optimizer.apply_gradients(zip(gradients, mlp.trainable_variables))
            avg_cost += loss / total_batch

        end_time = time.time()
        epoch_time = end_time - start_time
        training_metrics['epoch_times'].append(epoch_time)

        # Calculate validation accuracy
        validation_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(mlp(valid_features), axis=1), tf.argmax(valid_labels, axis=1)),
                    tf.float32)).numpy()
        training_metrics['epoch_losses'].append(avg_cost.numpy())
        training_metrics['validation_accuracies'].append(validation_accuracy)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{training_epochs}, Loss: {avg_cost:.4f}, Validation Accuracy: {validation_accuracy:.2%}, Time: {epoch_time:.2f}s")

    # Evaluate accuracy on the test set
    test_accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(mlp(test_features), axis=1), tf.argmax(test_labels, axis=1)), tf.float32)).numpy()
    training_metrics['test_accuracy'] = test_accuracy
    print(f"Test Accuracy for {n_layers} hidden layers: {test_accuracy:.2%}")

    # Append the metrics for the current configuration
    training_metrics_all_layers.append(training_metrics)

# Save metrics for all configurations
with open('deepnnscript_multiple_layers_metrics.pkl', 'wb') as f:
    pickle.dump(training_metrics_all_layers, f)

print("Metrics for all configurations saved to 'deepnnscript_multiple_layers_metrics.pkl'.")

# Plotting the metrics for different hidden layers
layer_counts = [metrics['n_layers'] for metrics in training_metrics_all_layers]
accuracies = [metrics['test_accuracy'] for metrics in training_metrics_all_layers]
training_times = [sum(metrics['epoch_times']) for metrics in training_metrics_all_layers]

plt.figure(figsize=(12, 5))

# Plot accuracy vs hidden layers
plt.subplot(1, 2, 1)
plt.plot(layer_counts, accuracies, marker='o', linestyle='-', color='blue', linewidth=2)
plt.xlabel('Number of Hidden Layers')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy vs. Number of Hidden Layers')
plt.grid(True)

# Plot training time vs hidden layers
plt.subplot(1, 2, 2)
plt.plot(layer_counts, training_times, marker='o', linestyle='-', color='blue', linewidth=2)
plt.xlabel('Number of Hidden Layers')
plt.ylabel('Total Training Time (Seconds)')
plt.title('Training Time vs. Number of Hidden Layers')
plt.grid(True)

plt.tight_layout()
plt.show()
