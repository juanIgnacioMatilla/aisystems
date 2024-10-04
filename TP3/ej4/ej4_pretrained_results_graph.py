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
                        'trained_models/ADAM_2E_784_5_5_10.pkl', 'trained_models/ADAM_2E_784_5_10.pkl']

    model_filenames3 = ['trained_models/ADAM_3E_784_64_10.pkl', 'trained_models/ADAM_3E_784_16_32_10.pkl',
                        'trained_models/ADAM_3E_784_16_10_10.pkl', 'trained_models/ADAM_3E_784_10_16_10.pkl',
                        'trained_models/ADAM_3E_784_10_10_10.pkl', 'trained_models/ADAM_3E_784_32_10.pkl',
                        'trained_models/ADAM_3E_784_10_10.pkl', 'trained_models/ADAM_3E_784_5_5_10.pkl',
                        'trained_models/ADAM_3E_784_5_10.pkl', 'trained_models/ADAM_3E_ReLU_784_10_10.pkl',
                        'trained_models/ADAM_3E_ReLU_784_32_10.pkl', 'trained_models/ADAM_3E_LeakyReLU_784_10_10.pkl',
                        'trained_models/ADAM_3E_LeakyReLU_784_32_10.pkl', 'trained_models/ADAM_3E_tanh_784_10_10.pkl',
                        'trained_models/ADAM_3E_tanh_784_32_10.pkl', 'trained_models/ADAM_3E_ELU_784_10_10.pkl',
                        'trained_models/ADAM_3E_ELU_784_32_10.pkl']

    model_filenames4 = ['trained_models/ADAM_4E_784_64_10.pkl', 'trained_models/ADAM_4E_784_10_10.pkl']

    model_filenames5 = ['trained_models/ADAM_5E_784_64_10.pkl', 'trained_models/ADAM_5E_784_10_10.pkl']

    model_filenames6 = ['trained_models/ADAM_6E_784_64_10.pkl', 'trained_models/ADAM_6E_784_10_10.pkl']

    model_filenames7 = ['trained_models/ADAM_7E_784_64_10.pkl']

    model_filenames = model_filenames1
    model_tags = []

    # Choose the list of tags selecting after 'adam' and before '.pkl'
    for model_filename in model_filenames:
        model_tags.append(model_filename.split('ADAM_')[1].split('_10.pkl')[0])

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

    # fig, ax = plt.subplots()
    # ax.plot(np.arange(len(model_filenames)), errors, label='Error')
    # ax.plot(np.arange(len(model_filenames)), accuracies, label='Accuracy')
    # ax.set_xticks(np.arange(len(model_filenames)))
    # ax.set_xticklabels(model_tags, rotation=45)
    # ax.legend()
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # ax.plot(np.arange(len(model_filenames)), times, label='Time')
    # ax.set_xticks(np.arange(len(model_filenames)))
    # ax.set_xticklabels(model_tags, rotation=45)
    # ax.legend()
    # plt.show()

    # overlay both graphs
    fig, ax1 = plt.subplots()

    # Plot errors and accuracies on ax1
    ax1.plot(np.arange(len(model_filenames)), errors, label='Error', color='blue')
    ax1.plot(np.arange(len(model_filenames)), accuracies, label='Accuracy', color='green')
    ax1.set_xticks(np.arange(len(model_filenames)))
    ax1.set_xticklabels(model_tags, rotation=45)
    ax1.set_ylabel('Error / Accuracy')
    ax1.legend(loc='upper left')

    # Create a second y-axis sharing the same x-axis, for the training times
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(model_filenames)), times, label='Time', color='red')
    ax2.set_ylabel('Time (s)')
    ax2.legend(loc='upper right')

    plt.title('Error, Accuracy, and Training Time Comparison')
    plt.tight_layout()
    plt.show()

    # plot accuracy divided by time
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(model_filenames)), np.array(accuracies) / np.array(times), label='Accuracy / Time')
    ax.set_xticks(np.arange(len(model_filenames)))
    ax.set_xticklabels(model_tags, rotation=45, ha="right")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
