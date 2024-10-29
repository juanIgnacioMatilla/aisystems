import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from TP4.src.model.boltzman.boltzmann_utils import load_mnist_data_split, store_model, load_model, add_noise_to_image, \
    load_mnist_data_split_sample, evaluate_model_ssim, calculate_ssim_statistics, plot_model_accuracies, extract_tags
from TP4.src.model.boltzman.deep_belief_network import DBN
from TP4.src.model.boltzman.number_plots import plot_all_reconstructions
from TP4.src.model.boltzman.restricted_boltzmann_machine import RBM

if __name__ == "__main__":
    # Cargar los datos con separación de entrenamiento y prueba
    x_train, y_train, x_test, y_test = load_mnist_data_split()
    n_samples_train, n_visible = x_train.shape
    n_samples_test, _ = x_test.shape

    # epochs = 5
    #
    # # Input layer with 784 units, one hidden layer
    # layer_sizes_1 = [784, 256]
    # layer_sizes_1_1 = [784, 512]
    # layer_sizes_1_2 = [784, 1024]
    # layer_sizes_1_3 = [784, 2048]
    #
    # # Input layer with 784 units, two hidden layers
    # layer_sizes_2 = [784, 128, 64]
    # layer_sizes_2_1 = [784, 256, 128]
    # layer_sizes_2_2 = [784, 512, 256]
    # layer_sizes_2_3 = [784, 1024, 512]
    #
    # # Input layer with 784 units, three hidden layers
    # layer_sizes_3 = [784, 256, 128, 64]
    # layer_sizes_3_1 = [784, 512, 256, 128]
    # layer_sizes_3_2 = [784, 1024, 512, 256]
    # layer_sizes_3_3 = [784, 2048, 1024, 512]
    #
    # # Input layer with 784 units, four hidden layers
    # layer_sizes_4 = [784, 256, 128, 64, 32]
    # layer_sizes_4_1 = [784, 512, 256, 128, 64]
    # layer_sizes_4_2 = [784, 1024, 512, 256, 128]
    # layer_sizes_4_3 = [784, 2048, 1024, 512, 256]
    #
    # dbn_1 = DBN(layer_sizes_1)
    # dbn_1_1 = DBN(layer_sizes_1_1)
    # dbn_1_2 = DBN(layer_sizes_1_2)
    # dbn_1_3 = DBN(layer_sizes_1_3)
    #
    # dbn_2 = DBN(layer_sizes_2)
    # dbn_2_1 = DBN(layer_sizes_2_1)
    # dbn_2_2 = DBN(layer_sizes_2_2)
    # dbn_2_3 = DBN(layer_sizes_2_3)
    #
    # dbn_3 = DBN(layer_sizes_3)
    # dbn_3_1 = DBN(layer_sizes_3_1)
    # dbn_3_2 = DBN(layer_sizes_3_2)
    # dbn_3_3 = DBN(layer_sizes_3_3)
    #
    # dbn_4 = DBN(layer_sizes_4)
    # dbn_4_1 = DBN(layer_sizes_4_1)
    # dbn_4_2 = DBN(layer_sizes_4_2)
    # dbn_4_3 = DBN(layer_sizes_4_3)
    #
    # dbn_1.pretrain(x_train, epochs=epochs, batch_size=100, learning_rate=0.01, k=1)
    # dbn_1_1.pretrain(x_train, epochs=epochs, batch_size=100, learning_rate=0.01, k=1)
    # dbn_1_2.pretrain(x_train, epochs=epochs, batch_size=100, learning_rate=0.01, k=1)
    # dbn_1_3.pretrain(x_train, epochs=epochs, batch_size=100, learning_rate=0.01, k=1)
    #
    # dbn_2.pretrain(x_train, epochs=epochs, batch_size=100, learning_rate=0.01, k=1)
    # dbn_2_1.pretrain(x_train, epochs=epochs, batch_size=100, learning_rate=0.01, k=1)
    # dbn_2_2.pretrain(x_train, epochs=epochs, batch_size=100, learning_rate=0.01, k=1)
    # dbn_2_3.pretrain(x_train, epochs=epochs, batch_size=100, learning_rate=0.01, k=1)
    #
    # dbn_3.pretrain(x_train, epochs=epochs, batch_size=100, learning_rate=0.01, k=1)
    # dbn_3_1.pretrain(x_train, epochs=epochs, batch_size=100, learning_rate=0.01, k=1)
    # dbn_3_2.pretrain(x_train, epochs=epochs, batch_size=100, learning_rate=0.01, k=1)
    # dbn_3_3.pretrain(x_train, epochs=epochs, batch_size=100, learning_rate=0.01, k=1)
    #
    # dbn_4.pretrain(x_train, epochs=epochs, batch_size=100, learning_rate=0.01, k=1)
    # dbn_4_1.pretrain(x_train, epochs=epochs, batch_size=100, learning_rate=0.01, k=1)
    # dbn_4_2.pretrain(x_train, epochs=epochs, batch_size=100, learning_rate=0.01, k=1)
    # dbn_4_3.pretrain(x_train, epochs=epochs, batch_size=100, learning_rate=0.01, k=1)
    #
    # store_model(dbn_1, 'trained_boltzmann/DBN_E5_784_256_B100_LR001_K1.pkl')
    # store_model(dbn_1_1, 'trained_boltzmann/DBN_E5_784_512_B100_LR001_K1.pkl')
    # store_model(dbn_1_2, 'trained_boltzmann/DBN_E5_784_1024_B100_LR001_K1.pkl')
    # store_model(dbn_1_3, 'trained_boltzmann/DBN_E5_784_2048_B100_LR001_K1.pkl')
    #
    # store_model(dbn_2, 'trained_boltzmann/DBN_E5_784_128_64_B100_LR001_K1.pkl')
    # store_model(dbn_2_1, 'trained_boltzmann/DBN_E5_784_256_128_B100_LR001_K1.pkl')
    # store_model(dbn_2_2, 'trained_boltzmann/DBN_E5_784_512_256_B100_LR001_K1.pkl')
    # store_model(dbn_2_3, 'trained_boltzmann/DBN_E5_784_1024_512_B100_LR001_K1.pkl')
    #
    # store_model(dbn_3, 'trained_boltzmann/DBN_E5_784_256_128_64_B100_LR001_K1.pkl')
    # store_model(dbn_3_1, 'trained_boltzmann/DBN_E5_784_512_256_128_B100_LR001_K1.pkl')
    # store_model(dbn_3_2, 'trained_boltzmann/DBN_E5_784_1024_512_256_B100_LR001_K1.pkl')
    # store_model(dbn_3_3, 'trained_boltzmann/DBN_E5_784_2048_1024_512_B100_LR001_K1.pkl')
    #
    # store_model(dbn_4, 'trained_boltzmann/DBN_E5_784_256_128_64_32_B100_LR001_K1.pkl')
    # store_model(dbn_4_1, 'trained_boltzmann/DBN_E5_784_512_256_128_64_B100_LR001_K1.pkl')
    # store_model(dbn_4_2, 'trained_boltzmann/DBN_E5_784_1024_512_256_128_B100_LR001_K1.pkl')
    # store_model(dbn_4_3, 'trained_boltzmann/DBN_E5_784_2048_1024_512_256_B100_LR001_K1.pkl')




    # model_filenamesDBN = [
    #     'trained_boltzmann/DBN_E1_784_500_200_60_B100_LR001_K1.pkl',
    #     'trained_boltzmann/DBN_E2_784_500_200_60_B100_LR001_K1.pkl',
    #     'trained_boltzmann/DBN_E3_784_500_200_60_B100_LR001_K1.pkl',
    #     'trained_boltzmann/DBN_E5_784_500_200_60_B100_LR001_K1.pkl',
    #     'trained_boltzmann/DBN_E8_784_500_200_60_B100_LR001_K1.pkl',
    #     'trained_boltzmann/DBN_E13_784_500_200_60_B100_LR001_K1.pkl',
    # ]
    #
    # model_filenamesRBM = [
    #     'trained_boltzmann/RBM_E1_784_60_B100_LR001_K1.pkl',
    #     'trained_boltzmann/RBM_E2_784_60_B100_LR001_K1.pkl',
    #     'trained_boltzmann/RBM_E3_784_60_B100_LR001_K1.pkl',
    #     'trained_boltzmann/RBM_E5_784_60_B100_LR001_K1.pkl',
    #     'trained_boltzmann/RBM_E8_784_60_B100_LR001_K1.pkl',
    #     'trained_boltzmann/RBM_E13_784_60_B100_LR001_K1.pkl'
    # ]
    #
    # model_filenames = model_filenamesRBM
    #
    # model_tags = extract_tags(model_filenames)
    #
    # # Initialize lists to store mean SSIM and standard deviation for each model
    # mean_ssims = []
    # std_ssims = []
    # num_runs = 3
    # noise_level = 0.1
    #
    # for model_filename in model_filenames:
    #     model = load_model(model_filename)
    #     print("Model loaded from file:", model_filename)
    #
    #     # Use the modularized function to get SSIM values for this model and noise level
    #     all_ssim_values = evaluate_model_ssim(model, x_test, noise_level, num_runs)
    #
    #     # Use the modularized function to calculate mean and std deviation of SSIM values
    #     mean_ssim, std_ssim = calculate_ssim_statistics(all_ssim_values)
    #
    #     # Append mean and std deviation for the model
    #     mean_ssims.append(mean_ssim)
    #     std_ssims.append(std_ssim)
    #
    #     # Print results for the current model
    #     model_tag = model_tags[model_filenames.index(model_filename)]
    #     print(f"Model: {model_tag} | Mean SSIM: {mean_ssim} | Std Dev: {std_ssim}")
    #
    # # Plot the results
    # plot_model_accuracies(model_tags, mean_ssims, std_ssims)