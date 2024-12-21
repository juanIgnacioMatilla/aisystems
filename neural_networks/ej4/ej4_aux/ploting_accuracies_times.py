import matplotlib.pyplot as plt
import numpy as np
from neural_networks.ej4.mnist_utils import print_training_errors, print_training_accuracies, load_model

def plot_times(times, labels):
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(labels))  # Create an array for the x positions

    # Create a bar plot for training times
    plt.bar(x_pos, times, align='center', alpha=0.7, color='blue')

    plt.title('Training Time for Different Optimization Algorithms')
    plt.xlabel('Models')
    plt.ylabel('Training Time (seconds)')
    plt.xticks(x_pos, labels)  # Set the x-tick labels to the model labels
    plt.ylim(0, max(times) + 10)  # Adjust y-axis to have enough space above max time
    plt.grid(axis='y')  # Add grid lines for better readability
    plt.show()

def plot_accuracies(epochs, all_accuracies, labels):
    plt.figure(figsize=(10, 6))
    for i, accuracies in enumerate(all_accuracies):
        plt.plot(epochs, accuracies, label=f'{labels[i]}')

    plt.title('Training Accuracies vs. Epochs for Different Optimization Algorithms')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)  # Accuracy is between 0 and 1
    plt.xlim(1, len(epochs))
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    optimization_comparation = ['trained_models/vanilla_batch20_10E_784_32_10.pkl',
                                'trained_models/vanilla_batch30_10E_784_32_10.pkl',
                                'trained_models/vanilla_batch60_10E_784_32_10.pkl',
                                ]

    labels = ['Batch size = 20', 'Batch size = 30', 'Batch size = 60']
    all_times = []
    all_accuracies = []
    epochs = None

    for model_filename in optimization_comparation:
        loaded_mlp = load_model(model_filename)
        print("Model loaded from file: ", model_filename)
        print("Time to train the model: ", loaded_mlp.training_time)

        # Capture total training time for each model
        all_times.append(loaded_mlp.training_time)

        if epochs is None:  # Set epochs if not already set
            epochs = np.arange(1, len(loaded_mlp.accuracies_by_epoch) + 1)  # Epochs offset by +1

        # Store accuracies for plotting
        all_accuracies.append(loaded_mlp.accuracies_by_epoch)

        # Print errors and accuracies for each model
        print_training_errors(loaded_mlp)
        print_training_accuracies(loaded_mlp)
        print()

    # Plot time as a bar plot
    plot_times(all_times, labels)

    # Plot accuracies vs. epochs
    plot_accuracies(epochs, all_accuracies, labels)

if __name__ == "__main__":
    main()
