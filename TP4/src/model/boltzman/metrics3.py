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


    layers = [784, 8096, 4096, 2048, 1024]
    model = DBN(layers, init_method='xavier')
    model.pretrain(x_train, epochs=100, batch_size=10, learning_rate=0.25, k=1)
    store_model(model, f'trained_boltzmann/DBN_E100_784_8096_4096_2048_1024_B10_LR25_K1.pkl')