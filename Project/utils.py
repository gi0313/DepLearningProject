import matplotlib.pyplot as plt
import torch
import random
import math
import os

os.makedirs('plots', exist_ok=True)

def plot_weight_images(weights_tensor, title_prefix="Unit", filename="weights.png"):
    #Create a 2x5 grid of subplots
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        #Reshaping the weights into a 32x32 image
        #Use .cpu().numpy() to safely move the data from PyTorch to Matplotlib
        img = weights_tensor[i].view(32, 32).cpu().numpy()

        # Plot using a grayscale colormap
        ax.imshow(img, cmap='gray')
        ax.set_title(f"{title_prefix} {i}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'plots/{filename}', bbox_inches='tight', dpi=300)
    plt.close()
    #plt.show()


def visualize_perceptron_weights(model, filename="perceptron_weights.png"):
    #Plots the weights of the Perceptron excluding the bias
    print("\nVisualizing Perceptron Weights...")
    #.weight.data grabs just the weights
    weights = model.fc.weight.data
    plot_weight_images(weights, title_prefix="Digit Class" , filename = filename)


def visualize_deep_nn_weights(model, is_regular_dnn=True, filename="dnn_weights.png"):
    #Extracts and plots weights for 10 units chosen at random
    print("\nVisualizing Deep NN First Hidden Layer Weights...")

    if is_regular_dnn:
        #RegularDeepNN uses nn.Sequential, so the first layer is at index 0
        first_layer_weights = model.network[0].weight.data
    else:
        #CustomDeepNN names the first layer 'fc1'
        first_layer_weights = model.fc1.weight.data

    total_units = first_layer_weights.size(0)

    #Choose 10 at random
    random_indices = random.sample(range(total_units), 10)

    #Extract just those 10 specific rows
    selected_weights = first_layer_weights[random_indices]
    plot_weight_images(selected_weights, title_prefix="Hidden Unit", filename=filename)


def plot_fitting_curves(history, filename="fitting_curves.png"):
    #Plots the training and test proxy error in one graph training and misclassification together on a separate graph.
    iterations = range(1, len(history['train_proxy']) + 1)

    #Create a figure with two side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    #Proxy Error graph Cross-Entropy Loss
    ax1.plot(iterations, history['train_proxy'], label='Training Proxy Error', color='blue')
    ax1.plot(iterations, history['test_proxy'], label='Test Proxy Error', color='orange', linestyle='--')
    ax1.set_title('Proxy Error (Half-MSE) vs. Iterations')
    ax1.set_xlabel('Number of Rounds (Iterations)')
    ax1.set_ylabel('Proxy Error')
    ax1.legend()
    ax1.grid(True)

    #Misclassification error graph Accuracy
    ax2.plot(iterations, history['train_misclass'], label='Training Misclassification', color='green')
    ax2.plot(iterations, history['test_misclass'], label='Test Misclassification', color='red', linestyle='--')
    ax2.set_title('Misclassification Rate vs. Iterations')
    ax2.set_xlabel('Number of Rounds (Iterations)')
    ax2.set_ylabel('Misclassification Rate (0.0 to 1.0)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'plots/{filename}', bbox_inches='tight', dpi=300)
    plt.close()
    #plt.show()

def plot_learning_curves(m_values, lc_data, filename="learning_curves.png"):
    #Learning curves plot
    #Create a figure with two side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    #Proxy Error vs training size
    ax1.plot(m_values, lc_data['train_proxy'], label='Train Proxy Error', marker='o', color='blue')
    ax1.plot(m_values, lc_data['test_proxy'], label='Test Proxy Error', marker='s', linestyle='--', color='orange')
    ax1.set_title('Learning Curve: Proxy Error vs. m')
    ax1.set_xlabel('Number of Training Examples (m)')
    ax1.set_ylabel('Proxy Error (Half-MSE)')
    ax1.legend()
    ax1.grid(True)

    #Misclassification Error vs m
    ax2.plot(m_values, lc_data['train_misclass'], label='Train Misclassification', marker='o', color='green')
    ax2.plot(m_values, lc_data['test_misclass'], label='Test Misclassification', marker='s', linestyle='--', color='red')
    ax2.set_title('Learning Curve: Misclassification Rate vs. m')
    ax2.set_xlabel('Number of Training Examples (m)')
    ax2.set_ylabel('Misclassification Rate (0.0 to 1.0)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'plots/{filename}', bbox_inches='tight', dpi=300)
    plt.close()
    #plt.show()


def calculate_T(layer_sizes):
    #Calculates computational complexity T(n) of the architecture based on layer sizes, as a list

    t_n = 0
    #Sum of (n_i + 1) * n_{i+1}
    for x in range(len(layer_sizes) - 1):
        t_n += (layer_sizes[x] + 1) * layer_sizes[x + 1]
    return t_n


def calculate_adaptive_R(m, layer_sizes, m_star, n_star, R_star):
    #Equation 1 R = min(1000, ceil( (m_star * T(n_star) / (m * T(n))) * R_star ))
    t_n = calculate_T(layer_sizes)
    t_n_star = calculate_T(n_star)

    #Formula
    ratio = (m_star * t_n_star) / (m * t_n)
    r_calculated = math.ceil(ratio * R_star)

    #1000 cap
    return min(1000, r_calculated)