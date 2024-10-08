from TP3.ej4.mnist_utils import print_training_errors, print_training_accuracies, load_model


def main():
    times = []
    errors = []
    accuracies = []

    # Load the trained model from a file
    model_filenames1 = ['trained_models/ADAM_1E_784_256_128_64_10.pkl', 'trained_models/ADAM_1E_784_128_10.pkl',
                        'trained_models/ADAM_1E_784_32_32_10.pkl', 'trained_models/ADAM_1E_784_64_10.pkl',
                        'trained_models/ADAM_1E_784_32_16_10.pkl', 'trained_models/ADAM_1E_784_32_10.pkl',
                        'trained_models/ADAM_1E_784_16_32_10.pkl', 'trained_models/ADAM_1E_784_16_10_10.pkl',
                        'trained_models/ADAM_1E_784_10_16_10.pkl', 'trained_models/ADAM_1E_784_10_10_10.pkl',
                        'trained_models/ADAM_1E_784_10_10.pkl', 'trained_models/ADAM_1E_784_5_5_10.pkl',
                        'trained_models/ADAM_1E_784_5_10.pkl']

    model_filenames2 = ['trained_models/ADAM_2E_784_128_10.pkl', 'trained_models/ADAM_2E_784_32_32_10.pkl',
                        'trained_models/ADAM_2E_784_64_10.pkl', 'trained_models/ADAM_2E_784_32_16_10.pkl',
                        'trained_models/ADAM_2E_784_32_10.pkl', 'trained_models/ADAM_2E_784_16_32_10.pkl',
                        'trained_models/ADAM_2E_784_16_10_10.pkl', 'trained_models/ADAM_2E_784_10_16_10.pkl',
                        'trained_models/ADAM_2E_784_10_10_10.pkl', 'trained_models/ADAM_2E_784_10_10.pkl',
                        'trained_models/ADAM_2E_784_5_5_10.pkl', 'trained_models/ADAM_2E_784_5_10.pkl'
                        ]

    model_filenames3 = ['trained_models/ADAM_3E_784_64_10.pkl', 'trained_models/ADAM_3E_784_16_32_10.pkl',
                        'trained_models/ADAM_3E_784_16_10_10.pkl', 'trained_models/ADAM_3E_784_10_16_10.pkl',
                        'trained_models/ADAM_3E_784_10_10_10.pkl', 'trained_models/ADAM_3E_784_32_10.pkl',
                        'trained_models/ADAM_3E_784_10_10.pkl', 'trained_models/ADAM_3E_784_5_5_10.pkl',
                        'trained_models/ADAM_3E_784_5_10.pkl', 'trained_models/ADAM_3E_ReLU_784_10_10.pkl',
                        'trained_models/ADAM_3E_ReLU_784_32_10.pkl', 'trained_models/ADAM_3E_LeakyReLU_784_10_10.pkl',
                        'trained_models/ADAM_3E_LeakyReLU_784_32_10.pkl', 'trained_models/ADAM_3E_tanh_784_10_10.pkl',
                        'trained_models/ADAM_3E_tanh_784_32_10.pkl', 'trained_models/ADAM_3E_ELU_784_10_10.pkl',
                        'trained_models/ADAM_3E_ELU_784_32_10.pkl']

    activation_functions32 = ['trained_models/ADAM_3E_784_32_10.pkl', 'trained_models/ADAM_3E_ReLU_784_32_10.pkl',
                              'trained_models/ADAM_3E_LeakyReLU_784_32_10.pkl',
                              'trained_models/ADAM_3E_tanh_784_32_10.pkl', 'trained_models/ADAM_3E_ELU_784_32_10.pkl']

    repeated_in123 = ['trained_models/ADAM_3E_784_128_10.pkl', 'trained_models/ADAM_3E_784_64_10.pkl',
                      'trained_models/ADAM_3E_784_32_32_10.pkl', 'trained_models/ADAM_3E_784_32_16_10.pkl',
                      'trained_models/ADAM_3E_784_16_32_10.pkl', 'trained_models/ADAM_3E_784_32_10.pkl',
                      'trained_models/ADAM_3E_784_16_10_10.pkl', 'trained_models/ADAM_3E_784_10_16_10.pkl',
                      'trained_models/ADAM_3E_784_10_10_10.pkl', 'trained_models/ADAM_3E_784_10_10.pkl',
                      'trained_models/ADAM_3E_784_5_5_10.pkl', 'trained_models/ADAM_3E_784_5_10.pkl']

    model_filenames4 = ['trained_models/ADAM_4E_784_64_10.pkl', 'trained_models/ADAM_4E_784_10_10.pkl']

    model_filenames5 = ['trained_models/ADAM_5E_784_64_10.pkl', 'trained_models/ADAM_5E_784_10_10.pkl']

    model_filenames6 = ['trained_models/ADAM_6E_784_64_10.pkl', 'trained_models/ADAM_6E_784_10_10.pkl']

    model_filenames7 = ['trained_models/ADAM_7E_784_64_10.pkl']

    other_model_filenames = ['trained_models/ADAM_10E_784_256_128_64_10.pkl',
                             'trained_models/ADAM_10E_784_16_10_10.pkl', 'trained_models/ADAM_100E_784_16_10_10.pkl',
                                'trained_models/OVERFITTED_VANILLA_1000E_784_10_10.pkl', 'trained_models/UNDERFITTED_VANILLA_10E_784_10_10.pkl',
                                'trained_models/UNDERFITTED_ADAM_1E_784_10_10.pkl', 'trained_models/OVERFITTED_ADAM_1000E_784_10_10.pkl']

    all_model_filenames = model_filenames1 + model_filenames2 + model_filenames3 + model_filenames4 + model_filenames5 + model_filenames6 + model_filenames7 + other_model_filenames
    model_filenames = other_model_filenames
    model_tags = []

    # Choose the list of tags selecting after 'adam' and before '.pkl'
    for model_filename in model_filenames:
        model_tags.append(model_filename.split('models/')[1].split('_10.pkl')[0])

    for model_filename in model_filenames:
        loaded_mlp = load_model(model_filename)
        times.append(loaded_mlp.training_time)
        errors.append(loaded_mlp.errors_by_epoch[-1])
        accuracies.append(loaded_mlp.accuracies_by_epoch[-1])
        print("Model loaded from file: ", model_filename)
        print("Time to train the model: ", loaded_mlp.training_time)
        print_training_errors(loaded_mlp)
        print_training_accuracies(loaded_mlp)
        print()

    # graph the results
    import matplotlib.pyplot as plt
    import numpy as np

    import matplotlib.pyplot as plt
    import numpy as np

    import numpy as np
    import matplotlib.pyplot as plt

    # Function to improve plotting
    def plot_with_enhancements(x_values, y_values, labels, title, ylabel, xlabel="Models", rotation=45):
        fig, ax = plt.subplots(figsize=(10, 6))  # Adjusted figure size for better readability

        # Create bar chart
        ax.bar(x_values, y_values, color=labels['line_color'])

        # Set xticks and labels
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_xticklabels(model_tags, rotation=rotation, ha="right", fontsize=10)

        # Add grid and labels
        ax.grid(axis='y', linestyle='--', alpha=0.7)  # Only horizontal grid lines
        ax.set_title(title, fontsize=14, pad=15)
        ax.set_ylabel(ylabel, labelpad=10, fontsize=12)
        ax.set_xlabel(xlabel, labelpad=10, fontsize=12)

        # Display legend
        ax.legend(loc='upper left', fontsize=10)

        # Improve layout
        plt.tight_layout()
        plt.show()

    # Individual Plots
    x_values = np.arange(len(model_filenames))

    # Accuracy Plot
    plot_with_enhancements(
        x_values, accuracies,
        labels={'line_color': 'green'},
        title='Model Accuracy Comparison', ylabel='Accuracy'
    )

    # Training Time Plot
    plot_with_enhancements(
        x_values, times,
        labels={'line_label': 'Training Time (s)', 'line_color': 'red'},
        title='Model Training Time Comparison', ylabel='Time (s)'
    )

    # # Overlay Plot for Error, Accuracy, and Training Time
    # def overlay_plots(x_values, errors, accuracies, times, model_tags):
    #     fig, ax1 = plt.subplots(figsize=(10, 6))
    #
    #     # Plot errors and accuracies on ax1
    #     ax1.plot(x_values, errors, label='Error', color='blue', marker='x')
    #     ax1.plot(x_values, accuracies, label='Accuracy', color='green', marker='o')
    #     ax1.set_xticks(np.arange(len(x_values)))
    #     ax1.set_xticklabels(model_tags, rotation=45, ha="right", fontsize=10)
    #     ax1.set_ylabel('Error / Accuracy', fontsize=12, labelpad=10)
    #     ax1.legend(loc='upper left', fontsize=10)
    #     ax1.grid(True, linestyle='--', alpha=0.7)
    #
    #     # Create a second y-axis sharing the same x-axis, for the training times
    #     ax2 = ax1.twinx()
    #     ax2.plot(x_values, times, label='Time (s)', color='red', marker='^')
    #     ax2.set_ylabel('Time (s)', fontsize=12, labelpad=10)
    #     ax2.legend(loc='upper right', fontsize=10)
    #
    #     # Add a title and improve layout
    #     plt.title('Error, Accuracy, and Training Time Comparison', fontsize=14, pad=15)
    #     plt.tight_layout()
    #     plt.show()
    #
    # # Overlay plot
    # overlay_plots(x_values, errors, accuracies, times, model_tags)
    #
    # # Plot accuracy divided by time
    # plot_with_enhancements(
    #     x_values, np.array(accuracies) / np.array(times),
    #     labels={'line_label': 'Accuracy / Time', 'line_color': 'purple'},
    #     title='Accuracy per Time Ratio', ylabel='Accuracy / Time'
    # )

    # # overlay both graphs
    # fig, ax1 = plt.subplots()
    #
    # # Plot errors and accuracies on ax1
    # ax1.plot(np.arange(len(model_filenames)), errors, label='Error', color='blue')
    # ax1.plot(np.arange(len(model_filenames)), accuracies, label='Accuracy', color='green')
    # ax1.set_xticks(np.arange(len(model_filenames)))
    # ax1.set_xticklabels(model_tags, rotation=45)
    # ax1.set_ylabel('Error / Accuracy')
    # ax1.legend(loc='upper left')
    #
    # # Create a second y-axis sharing the same x-axis, for the training times
    # ax2 = ax1.twinx()
    # ax2.plot(np.arange(len(model_filenames)), times, label='Time', color='red')
    # ax2.set_ylabel('Time (s)')
    # ax2.legend(loc='upper right')
    #
    # plt.title('Error, Accuracy, and Training Time Comparison')
    # plt.tight_layout()
    # plt.show()
    #
    # # plot accuracy divided by time
    # fig, ax = plt.subplots()
    # ax.plot(np.arange(len(model_filenames)), np.array(accuracies) / np.array(times), label='Accuracy / Time')
    # ax.set_xticks(np.arange(len(model_filenames)))
    # ax.set_xticklabels(model_tags, rotation=45, ha="right")
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
