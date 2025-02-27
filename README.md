# Neural Network Hyperparameter Optimization

## Project Overview
This project optimizes neural network hyperparameters (hidden units & regularization λ) for accuracy, generalization, and efficiency. Using MNIST & CelebA datasets, we compare NN, DNN, and CNN models. Results show **12-20 hidden units with λ = 15-25** balance performance. CNNs excel in image classification. Includes training, evaluation, and visualization code.

## Key Features
- **Hyperparameter Tuning**: Tests various hidden unit counts (4, 8, 12, 16, 20) and λ values (0 to 60).
- **Overfitting & Underfitting Analysis**: Identifies best model configurations.
- **Performance Comparison**:
  - Neural Networks (NN)
  - Deep Neural Networks (DNN)
  - Convolutional Neural Networks (CNN)
- **Training Efficiency**: Evaluates training time vs. accuracy trade-offs.
- **Graphical Insights**: Heatmaps and accuracy plots visualize results.

## Results Summary

### **MNIST Dataset**
- **Best NN Model**:
  - **Hidden Units:** 20
  - **Regularization (λ):** 10
  - **Test Accuracy:** 93.66%
- **Best CNN Model**:
  - **Test Accuracy:** 98.6% (after 10,000 iterations)

### **CelebA Dataset**
- **Best NN Model**:
  - **Hidden Units:** 50
  - **Regularization (λ):** 95
  - **Test Accuracy:** 86.75%
- **Best Deep NN Model**:
  - **1 Hidden Layer:**  
    - **Test Accuracy:** 82.59%  
    - **Training Time:** 4.62 sec/epoch

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
