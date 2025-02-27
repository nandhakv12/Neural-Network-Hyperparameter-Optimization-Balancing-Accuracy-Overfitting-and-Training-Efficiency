import pickle

# Load the best model parameters from the pickle file
with open('params.pickle', 'rb') as f:
    params = pickle.load(f)

# Extract and print the parameters
optimal_n_hidden = params['optimal_n_hidden']
optimal_lambda = params['optimal_lambda']
w1 = params['w1']
w2 = params['w2']
selected_features = params['selected_features']

print("Best Model Parameters:")
print(f"Optimal Number of Hidden Units: {optimal_n_hidden}")
print(f"Optimal Regularization λ: {optimal_lambda}")

# Optionally, print the shapes of the weight matrices to verify their dimensions
print(f"Shape of Weights between Input and Hidden Layer (w1): {w1.shape}")
print(f"Shape of Weights between Hidden and Output Layer (w2): {w2.shape}")

# Print selected features indices
print(f"Number of Selected Features: {len(selected_features)}")
print(f"Indices of Selected Features: {selected_features}")
