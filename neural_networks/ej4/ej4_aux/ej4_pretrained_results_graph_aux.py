from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from neural_networks.ej4.mnist_utils import print_training_errors, print_training_accuracies, load_model


def main():
    times = defaultdict(list)
    errors = defaultdict(list)
    accuracies = defaultdict(list)
    neuron_counts = {}  # Store the total number of neurons for each model

    repeated_in123 = (['trained_models/ADAM_3E_784_128_10.pkl', 'trained_models/1_ADAM_3E_784_128_10.pkl', 'trained_models/2_ADAM_3E_784_128_10.pkl',
                      'trained_models/ADAM_3E_784_64_10.pkl', 'trained_models/1_ADAM_3E_784_64_10.pkl', 'trained_models/2_ADAM_3E_784_64_10.pkl',
                      'trained_models/ADAM_3E_784_32_32_10.pkl', 'trained_models/1_ADAM_3E_784_32_32_10.pkl', 'trained_models/2_ADAM_3E_784_32_32_10.pkl',
                      'trained_models/ADAM_3E_784_32_16_10.pkl', 'trained_models/1_ADAM_3E_784_32_16_10.pkl', 'trained_models/2_ADAM_3E_784_32_16_10.pkl',
                      'trained_models/ADAM_3E_784_16_32_10.pkl', 'trained_models/1_ADAM_3E_784_16_32_10.pkl', 'trained_models/2_ADAM_3E_784_16_32_10.pkl',
                      'trained_models/ADAM_3E_784_32_10.pkl', 'trained_models/1_ADAM_3E_784_32_10.pkl', 'trained_models/2_ADAM_3E_784_32_10.pkl',
                      'trained_models/ADAM_3E_784_16_10_10.pkl', 'trained_models/1_ADAM_3E_784_16_10_10.pkl', 'trained_models/2_ADAM_3E_784_16_10_10.pkl',
                      'trained_models/ADAM_3E_784_10_16_10.pkl',    'trained_models/1_ADAM_3E_784_10_16_10.pkl', 'trained_models/2_ADAM_3E_784_10_16_10.pkl',
                      'trained_models/ADAM_3E_784_10_10_10.pkl', 'trained_models/1_ADAM_3E_784_10_10_10.pkl', 'trained_models/2_ADAM_3E_784_10_10_10.pkl',
                       'trained_models/ADAM_3E_784_10_5_10.pkl', 'trained_models/1_ADAM_3E_784_10_5_10.pkl', 'trained_models/2_ADAM_3E_784_10_5_10.pkl',
                      'trained_models/ADAM_3E_784_10_10.pkl', 'trained_models/1_ADAM_3E_784_10_10.pkl', 'trained_models/2_ADAM_3E_784_10_10.pkl',
                      'trained_models/ADAM_3E_784_5_5_10.pkl', 'trained_models/1_ADAM_3E_784_5_5_10.pkl', 'trained_models/2_ADAM_3E_784_5_5_10.pkl',
                      'trained_models/ADAM_3E_784_5_10.pkl', 'trained_models/1_ADAM_3E_784_5_10.pkl', 'trained_models/2_ADAM_3E_784_5_10.pkl'])


    activation_functions32 = [
        'trained_models/ADAM_3E_ELU_784_32_10.pkl', 'trained_models/1_ADAM_3E_ELU_784_32_10.pkl', 'trained_models/2_ADAM_3E_ELU_784_32_10.pkl',
                            'trained_models/ADAM_3E_LeakyReLU_784_32_10.pkl', 'trained_models/1_ADAM_3E_LeakyReLU_784_32_10.pkl', 'trained_models/2_ADAM_3E_LeakyReLU_784_32_10.pkl',
                            'trained_models/ADAM_3E_tanh_784_32_10.pkl', 'trained_models/1_ADAM_3E_tanh_784_32_10.pkl', 'trained_models/2_ADAM_3E_tanh_784_32_10.pkl',
    'trained_models/ADAM_3E_Sigmoid_784_32_10.pkl', 'trained_models/1_ADAM_3E_Sigmoid_784_32_10.pkl', 'trained_models/2_ADAM_3E_Sigmoid_784_32_10.pkl']

    model_tags = []  # Use tags to differentiate the structures (e.g., '784_128', '784_64', etc.)

    for model_filename in activation_functions32:
        # Extract tag for each model (e.g., '784_128' from 'ADAM_3E_784_128_10.pkl')
        tag = model_filename.split('ADAM_3E_')[1].split('_10.pkl')[0]
        if tag not in model_tags:
            model_tags.append(tag)

        # Load the model and store results in the corresponding tag group
        loaded_mlp = load_model(model_filename)
        times[tag].append(loaded_mlp.training_time)
        errors[tag].append(loaded_mlp.errors_by_epoch[-1])
        accuracies[tag].append(loaded_mlp.accuracies_by_epoch[-1])
        print_training_errors(loaded_mlp)
        print_training_accuracies(loaded_mlp)

    # Compute mean and standard deviation for each tag
    mean_times = [np.mean(times[tag]) for tag in model_tags]
    stdev_times = [np.std(times[tag]) for tag in model_tags]

    mean_accuracies = [np.mean(accuracies[tag]) for tag in model_tags]
    stdev_accuracies = [np.std(accuracies[tag]) for tag in model_tags]

    # Function to improve plotting with error bars
    def plot_with_enhancements(x_values, y_values, y_errors, labels, title, ylabel, xlabel="Models", rotation=45,
                               ylim=None):
        fig, ax = plt.subplots(figsize=(10, 6))  # Adjusted figure size for better readability

        # Create bar chart with error bars
        ax.bar(x_values, y_values, yerr=y_errors, color=labels['line_color'], capsize=5)

        # Set xticks and labels
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_xticklabels(model_tags, rotation=rotation, ha="right", fontsize=10)

        # Add grid and labels
        ax.grid(axis='y', linestyle='--', alpha=0.7)  # Only horizontal grid lines
        ax.set_title(title, fontsize=14, pad=15)
        ax.set_ylabel(ylabel, labelpad=10, fontsize=12)
        ax.set_xlabel(xlabel, labelpad=10, fontsize=12)

        # Set y-axis limits if specified
        if ylim:
            ax.set_ylim(ylim)

        # Improve layout
        plt.tight_layout()
        plt.show()

    # Individual Plots
    x_values = np.arange(len(model_tags))

    # Accuracy Plot with Error Bars
    plot_with_enhancements(
        x_values, mean_accuracies, stdev_accuracies,
        labels={'line_label': 'Accuracy', 'line_color': 'green'},
        title='Model Accuracy Comparison - 3 runs', ylabel='Accuracy', ylim=(0.7, 1)
    )

    # Training Time Plot with Error Bars
    plot_with_enhancements(
        x_values, mean_times, stdev_times,
        labels={'line_label': 'Training Time (s)', 'line_color': 'red'},
        title='Model Training Time Comparison - 3 runs', ylabel='Time (s)'
    )


    # # Training Time vs. Number of Neurons
    # neuron_values = [neuron_counts[tag] for tag in model_tags]
    #
    # # Scatter plot for time vs. number of neurons
    # plt.figure(figsize=(10, 6))
    # plt.scatter(neuron_values, mean_times, color='blue')
    # plt.errorbar(neuron_values, mean_times, yerr=stdev_times, fmt='o', color='blue', capsize=5)
    # plt.title('Training Time vs. Number of Inner Neurons', fontsize=14)
    # plt.xlabel('Number of Inner Neurons', fontsize=12)
    # plt.ylabel('Training Time (s)', fontsize=12)
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()
