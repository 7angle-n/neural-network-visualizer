# **Neural Network Playground using TensorFlow + Streamlit**
An interactive web app built with Streamlit and TensorFlow that lets you experiment with neural network architectures, activation functions, and synthetic datasets in real time. Visualize decision boundaries, training metrics, and even the network graph itself—all without writing a single line of code. 

## Features
1. Dataset Selection: Choose from moons, circles, or XOR—classic synthetic datasets for classification.
2. Custom Architecture: Define hidden layer sizes and activation functions layer-by-layer.
3. Live Training: Train models instantly with adjustable hyperparameters (learning rate, epochs, batch size).
4. Decision Boundary Plot: See how your model classifies the input space.
5. Training Metrics: Track loss, accuracy, and validation accuracy over epochs.
6. Network Visualization: View a color-coded graph of your neural network ar6chitecture with activation legends.

## Demo
<img width="1919" height="872" alt="Screenshot 2025-11-13 182756" src="https://github.com/user-attachments/assets/6f3f9fc9-6528-4875-823d-b4e7c7a91f6a" />
<img width="1919" height="874" alt="Screenshot 2025-11-13 182739" src="https://github.com/user-attachments/assets/9851fdc1-8231-45bc-83bd-bf455c6ba6e6" />

## Visual Outputs
1. Decision Boundary: Shows how the trained model separates classes.
2. Training Metrics: Plots loss, accuracy, and validation accuracy.
3. Network Graph: Node-link diagram with activation-based coloring.

## How It Works
Data Generation: Uses scikit-learn or custom logic for XOR.
Model Building: Constructs a Sequential model with user-defined layers and activations.
Training: Uses Adam optimizer and sparse_categorical_crossentropy loss.
Visualization:
  Decision boundary via meshgrid prediction.
  Training metrics via matplotlib.
  Network graph via networkx.
