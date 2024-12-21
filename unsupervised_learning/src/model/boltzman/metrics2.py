import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from TP4.src.model.boltzman.boltzmann_utils import load_mnist_data_split, store_model, load_model, add_noise_to_image, \
    load_mnist_data_split_sample, evaluate_model_ssim, calculate_ssim_statistics, plot_model_accuracies, extract_tags, \
    extract_ts_tags, extract_hidden_units_tags, extract_intermediate_tags, plot_model_times, extract_weight_init_tags, \
    extract_lr_tags, plot_basic_metrics, plot_accuracy_vs_noise, extract_tags_up_to_b, extract_bs_tags
from TP4.src.model.boltzman.deep_belief_network import DBN
from TP4.src.model.boltzman.number_plots import plot_all_reconstructions
from TP4.src.model.boltzman.restricted_boltzmann_machine import RBM

if __name__ == "__main__":
    # # Cargar los datos con separación de entrenamiento y prueba
    x_train, y_train, x_test, y_test = load_mnist_data_split()
    n_samples_train, n_visible = x_train.shape
    n_samples_test, _ = x_test.shape


    # train the model with different batch sizes
    model = RBM(n_visible=784, n_hidden=8096, init_method='xavier')
    model.train(x_train, epochs=100, batch_size=10, learning_rate=0.25, k=1)
    store_model(model, f'trained_boltzmann/RBM_E100_784_8096_B10_LR25_K1.pkl')

    # layers = [784, 8096, 4096, 2048, 1024]
    # model = DBN(layers, init_method='xavier')
    # model.pretrain(x_train, epochs=100, batch_size=10, learning_rate=0.25, k=1)
    # store_model(model, f'trained_boltzmann/DBN_E100_784_8096_4096_2048_1024_B10_LR25_K1.pkl')

    model_filenamesDBN = [
        'trained_boltzmann/DBN_E1_784_500_200_60_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E2_784_500_200_60_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E3_784_500_200_60_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_500_200_60_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E8_784_500_200_60_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E13_784_500_200_60_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E21_784_500_200_60_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E34_784_500_200_60_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E55_784_500_200_60_B100_LR001_K1.pkl'
    ]
    model_filenamesRBM = [
        'trained_boltzmann/RBM_E1_784_60_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E2_784_60_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E3_784_60_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_60_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E8_784_60_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E13_784_60_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E21_784_60_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E34_784_60_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E55_784_60_B100_LR001_K1.pkl'
    ]

    model_filenamesDBN_TS = [
        'trained_boltzmann/DBN_E8_784_500_200_60_B100_LR001_K1_TS1000.pkl',
        'trained_boltzmann/DBN_E8_784_500_200_60_B100_LR001_K1_TS5000.pkl',
        'trained_boltzmann/DBN_E8_784_500_200_60_B100_LR001_K1_TS10000.pkl',
        'trained_boltzmann/DBN_E8_784_500_200_60_B100_LR001_K1_TS15000.pkl',
        'trained_boltzmann/DBN_E8_784_500_200_60_B100_LR001_K1_TS20000.pkl',
        'trained_boltzmann/DBN_E8_784_500_200_60_B100_LR001_K1_TS25000.pkl',
        'trained_boltzmann/DBN_E8_784_500_200_60_B100_LR001_K1_TS30000.pkl',
        'trained_boltzmann/DBN_E8_784_500_200_60_B100_LR001_K1_TS35000.pkl',
        'trained_boltzmann/DBN_E8_784_500_200_60_B100_LR001_K1_TS40000.pkl',
        'trained_boltzmann/DBN_E8_784_500_200_60_B100_LR001_K1_TS45000.pkl',
        'trained_boltzmann/DBN_E8_784_500_200_60_B100_LR001_K1_TS50000.pkl',
        'trained_boltzmann/DBN_E8_784_500_200_60_B100_LR001_K1_TS55000.pkl',
        'trained_boltzmann/DBN_E8_784_500_200_60_B100_LR001_K1.pkl'
    ]
    model_filenamesRBM_TS = [
        'trained_boltzmann/RBM_E8_784_60_B100_LR001_K1_TS5000.pkl',
        'trained_boltzmann/RBM_E8_784_60_B100_LR001_K1_TS10000.pkl',
        'trained_boltzmann/RBM_E8_784_60_B100_LR001_K1_TS15000.pkl',
        'trained_boltzmann/RBM_E8_784_60_B100_LR001_K1_TS20000.pkl',
        'trained_boltzmann/RBM_E8_784_60_B100_LR001_K1_TS25000.pkl',
        'trained_boltzmann/RBM_E8_784_60_B100_LR001_K1_TS30000.pkl',
        'trained_boltzmann/RBM_E8_784_60_B100_LR001_K1_TS35000.pkl',
        'trained_boltzmann/RBM_E8_784_60_B100_LR001_K1_TS40000.pkl',
        'trained_boltzmann/RBM_E8_784_60_B100_LR001_K1_TS45000.pkl',
        'trained_boltzmann/RBM_E8_784_60_B100_LR001_K1_TS50000.pkl',
        'trained_boltzmann/RBM_E8_784_60_B100_LR001_K1_TS55000.pkl',
        'trained_boltzmann/RBM_E8_784_60_B100_LR001_K1.pkl'
    ]

    model_filenamesRBM_E5 = [
        'trained_boltzmann/RBM_E5_784_16_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_32_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_64_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_128_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_256_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_512_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_784_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_1024_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_2048_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_4096_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_8192_B100_LR001_K1.pkl'
    ]
    model_filenamesDBN_E5 = [
        'trained_boltzmann/DBN_E5_784_128_64_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_256_128_B100_LR001_K1.pkl',
        # 'trained_boltzmann/DBN_E5_784_256_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_500_200_60_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_512_256_B100_LR001_K1.pkl',
        # 'trained_boltzmann/DBN_E5_784_512_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_512_256_128_64_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_512_256_128_B100_LR001_K1.pkl',
        # 'trained_boltzmann/DBN_E5_784_1024_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_1024_512_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_1024_512_256_128_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_1024_512_256_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_256_128_64_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_256_128_64_32_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_2048_1024_512_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_2048_1024_512_256_B100_LR001_K1.pkl',
        # 'trained_boltzmann/DBN_E5_784_2048_B100_LR001_K1.pkl',
        # 'trained_boltzmann/DBN_E5_784_4096_B100_LR001_K1.pkl',
        # 'trained_boltzmann/DBN_E5_784_8192_B100_LR001_K1.pkl'
    ]

    model_filenamesDBN_2048 = [
        'trained_boltzmann/DBN_E5_784_2048_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_2048_2048_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_2048_2048_2048_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_2048_2048_2048_2048_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_2048_2048_2048_2048_2048_B100_LR001_K1.pkl'
    ]

    model_filenamesDBN_WeightInit = [
        'trained_boltzmann/XAVIER_DBN_E5_784_2048_B100_LR001_K1.pkl',
        'trained_boltzmann/HE_DBN_E5_784_2048_B100_LR001_K1.pkl',
        'trained_boltzmann/UNIFORM_DBN_E5_784_2048_B100_LR001_K1.pkl',
        'trained_boltzmann/NORMAL_DBN_E5_784_2048_B100_LR001_K1.pkl'
    ]
    model_filenamesRBM_WeightInit = [
        'trained_boltzmann/XAVIER_RBM_E5_784_64_B100_LR001_K1.pkl',
        'trained_boltzmann/HE_RBM_E5_784_64_B100_LR001_K1.pkl',
        'trained_boltzmann/UNIFORM_RBM_E5_784_64_B100_LR001_K1.pkl',
        'trained_boltzmann/NORMAL_RBM_E5_784_64_B100_LR001_K1.pkl'
    ]

    model_filenamesRBM_LR = [
        'trained_boltzmann/RBM_E5_784_128_B100_LR05_K1.pkl',
        'trained_boltzmann/RBM_E5_784_128_B100_LR1_K1.pkl',
        'trained_boltzmann/RBM_E5_784_128_B100_LR15_K1.pkl',
        'trained_boltzmann/RBM_E5_784_128_B100_LR2_K1.pkl',
        'trained_boltzmann/RBM_E5_784_128_B100_LR25_K1.pkl',
        'trained_boltzmann/RBM_E5_784_128_B100_LR3_K1.pkl',
        'trained_boltzmann/RBM_E5_784_128_B100_LR35_K1.pkl',
        'trained_boltzmann/RBM_E5_784_128_B100_LR4_K1.pkl',
        'trained_boltzmann/RBM_E5_784_128_B100_LR45_K1.pkl',
        'trained_boltzmann/RBM_E5_784_128_B100_LR5_K1.pkl'
    ]

    model_filenamesRBM_v_DBN1 = [
        'trained_boltzmann/RBM_E5_784_16_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_32_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_64_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_128_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_256_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_8_8_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_16_16_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_32_32_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_64_64_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_128_128_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_8_4_4_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_16_8_8_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_32_16_16_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_64_32_32_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_128_64_64_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_8_4_2_2_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_16_8_4_4_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_32_16_8_8_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_64_32_16_16_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_128_64_32_32_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_8_4_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_16_8_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_32_16_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_64_32_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_128_64_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_8_4_2_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_16_8_4_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_32_16_8_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_64_32_16_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_128_64_32_B100_LR001_K1.pkl'
    ]
    model_filenamesRBM_v_DBN2 = [
        'trained_boltzmann/DBN_E5_784_128_64_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_256_128_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_256_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_500_200_60_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_512_256_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_512_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_512_256_128_64_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_512_256_128_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_1024_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_1024_512_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_1024_512_256_128_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_1024_512_256_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_256_128_64_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_256_128_64_32_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_2048_1024_512_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_2048_1024_512_256_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_2048_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_4096_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_8192_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_256_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_512_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_1024_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_2048_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_4096_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_8192_B100_LR001_K1.pkl'
    ]

    model_filenamesRBM_v_DBN1_CLEAN = [
        'trained_boltzmann/RBM_E5_784_16_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_32_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_64_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_128_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_256_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_32_32_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_8_4_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_64_32_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_128_64_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_16_8_4_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_32_16_8_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_64_32_16_B100_LR001_K1.pkl',
    ]
    model_filenamesRBM_v_DBN2_CLEAN = [
        'trained_boltzmann/DBN_E5_784_256_128_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_512_256_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_512_256_128_64_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_512_256_128_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_1024_512_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_1024_512_256_128_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_1024_512_256_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_2048_1024_512_B100_LR001_K1.pkl',
        'trained_boltzmann/DBN_E5_784_2048_1024_512_256_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_512_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_1024_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_2048_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_4096_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_8192_B100_LR001_K1.pkl'
    ]

    model_filenamesRBM_BS = [
        'trained_boltzmann/RBM_E5_784_64_B1_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_64_B5_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_64_B10_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_64_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_64_B1000_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_64_B5000_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_64_B10000_LR001_K1.pkl',
        'trained_boltzmann/RBM_E5_784_64_B30000_LR001_K1.pkl'
    ]

    best_models = [
        'trained_boltzmann/DBN_E55_784_500_200_60_B100_LR001_K1.pkl',
        'trained_boltzmann/RBM_E55_784_60_B100_LR001_K1.pkl'
    ]

    # model_filenames = best_models
    # model_tags = []
    # for model_filename in model_filenames:
    #     complete_tag = model_filename.split('/')
    #     model_tag = complete_tag[1]
    #     model_tags.append(model_tag)
    #
    # # test the model with different my mlp
    # mlp_test_model = load_model('../../../../neural_networks/ej4/trained_models/XAVIER_ADAM_10E_784_256_128_64_10.pkl')
    #
    # for noise_level in np.arange(0, 0.3, 0.05):
    #     accuracy_by_model = []
    #
    #     for model_filename in model_filenames:
    #         results = []
    #         model = load_model(model_filename)
    #         # test with the whole test set
    #         for x_test_sample, y_test_sample in zip(x_test, y_test):
    #             #add noise to the sample
    #             noisy_sample = add_noise_to_image(x_test_sample, noise_level)
    #             prediction = np.argmax(mlp_test_model.predict(model.reconstruct(noisy_sample).reshape(1, -1)))
    #             results.append(prediction == y_test_sample)
    #
    #         accuracy = np.mean(results)
    #         accuracy_by_model.append(accuracy)
    #
    #     print(f"Results by model: {accuracy_by_model}")
    #
    #     # plot accuracy and show the accuracy result number inside the bar
    #     #show between 0.8 and 1 with 0.01 step
    #     plt.figure(figsize=(12, 10))
    #     plt.title("Accuracy by model - Noise: " + str(noise_level))
    #     plt.xticks(rotation=45)
    #     plt.bar(model_tags, accuracy_by_model, color='green', edgecolor='grey')
    #     for i, v in enumerate(accuracy_by_model):
    #         plt.text(i, v + 0.01, f"{v:.4f}", ha='center', va='bottom')
    #     plt.xlabel("Model")
    #     plt.ylabel("Accuracy")
    #     plt.show()
