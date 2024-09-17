import math
from typing import Tuple, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
from TP2.src.model.chromosome import Chromosome
from TP2.src.model.individual import Individual

# Función que grafica el radar chart para una instancia de Chromosome
# Función que grafica el radar chart para una instancia de Chromosome
def plot_chromosome_distribution(chromosome: Chromosome, type: str):
    # Obtener los puntos de cada atributo
    values = [
        chromosome.strength_points(),
        chromosome.dexterity_points(),
        chromosome.intelligence_points(),
        chromosome.constitution_points(),
        chromosome.vitality_points()
    ]
    df = pd.DataFrame(dict(
        r= values,
        theta=['fuerza', 'destreza', 'inteligencia',
               'constitucion', 'vigor']))
    # Cambiar el rango del eje radial y los valores de los ticks

    fig = px.line_polar(df, r='r', theta='theta', line_close=True)
    # Cambiar el rango del eje radial, ajustar el tamaño de la fuente de los labels y números
    fig.update_layout(
        font=dict(
            size=18  
        )
    )
    fig.write_image(f"grafico_polar{type}.png")
    print("saved image")
    fig.show()
def plot_diversity_per_generation(diversity_history_runs: List[List[Tuple[float, float]]], best_inds: List[Individual],best_gens:List[int],
                                  config):
    means, std_devs = compute_means_and_stds(diversity_history_runs)
    x = np.arange(1, len(means) * 25, 25)

    # # Filtrar los valores cada 25 elementos
    # x = x[::20]
    # means = means[::20]
    # std_devs = std_devs[::20]

    # Crear figura y subplots (2 columnas: 70% gráfico, 30% texto)
    fig, ax = plt.subplots(figsize=(10, 6), nrows=1, ncols=2, gridspec_kw={'width_ratios': [3, 1]})

    # Gráfico de barras de error en el subplot izquierdo
    ax[0].errorbar(x, means, yerr=std_devs, fmt='o', capsize=5, linestyle='None', marker='o', color='b', ecolor='r')

    # Agregar etiquetas y título
    ax[0].set_xlabel('Generaciones')
    ax[0].set_ylabel('Diversidad')
    ax[0].set_title('Diversidad por Generación')

    # # Configurar las marcas en el eje x para que comiencen en 1 y luego marquen cada 10
    # ax[0].set_xticks(np.concatenate([[1], np.arange(50, len(means) + 1, 50)]))

    # Mostrar cuadrícula en el gráfico
    ax[0].grid(True)
    best_fitness = [ind.fitness() for ind in best_inds]
    best_fitness_mean = np.mean(best_fitness)
    best_fitness_std = np.std(best_fitness)

    best_gens_mean = np.mean(best_gens)
    best_gens_std = np.std(best_gens)
    # Texto de hiperparámetros en el subplot derecho
    hyperparameters = {
        'Mejor fitness': f'{best_fitness_mean:.3f} ± {best_fitness_std:.3f}',
        'Generacion del mejor fitness': f'{best_gens_mean:.3f} ± {best_gens_std:.3f}',
        'Condición de corte': f'{config['hyperparams']['termination']['amount']} gen',
        'Tiempo máximo': f'{config['time_limit']}s',
        'Selección 1': 'Torneo probabilistico 80%',
        'Threshold': '0.6',
        'Selección 2': 'Roulette 20%',
        'K': f"{config['hyperparams']['selection']['k']}",
        'Mutación': 'Multi Gene',
        'P(mutación)': '5%',
        'Cruza': 'Uniforme',
        'Reemplazo': 'FillAll',
        'Selección 1 de reemplazo': 'Torneo probabilistico 80%',
        'Selección 2 de reemplazo': 'Ranking 20%',
        'Puntos totales': f'{config['total_points']}',
        'Cantidad Runs': f'{config['runs']}',
        'Tamaño de poblacion': f'{config["population_size"]}',
        'Tipo': f'Mago',
    }

    # Alinear el texto a la izquierda dentro del subplot derecho
    hyperparams_text = "\n".join([f"{param}: {value}" for param, value in hyperparameters.items()])
    ax[1].text(0, 1, hyperparams_text, transform=ax[1].transAxes, fontsize=12, verticalalignment='top',
               horizontalalignment='left')

    # Quitar el eje de la columna del texto
    ax[1].axis('off')

    # Ajustar el layout para que no se superpongan
    plt.tight_layout()
    plt.show()


def plot_fitness_per_generation(fitness_history_runs: List[List[Tuple[float, float]]], config):
    means, std_devs = compute_means_and_stds(fitness_history_runs)
    x = np.arange(1, (len(means) + 1))

    # Filtrar los valores cada 25 elementos
    x = x[::20]
    means = means[::20]
    std_devs = std_devs[::20]

    # Crear figura y subplots (2 columnas: 70% gráfico, 30% texto)
    fig, ax = plt.subplots(figsize=(10, 6), nrows=1, ncols=2, gridspec_kw={'width_ratios': [3, 1]})

    # Gráfico de barras de error en el subplot izquierdo
    ax[0].errorbar(x, means, yerr=std_devs, fmt='o', capsize=5, linestyle='None', marker='o', color='b', ecolor='r')

    # Agregar etiquetas y título
    ax[0].set_xlabel('Generaciones')
    ax[0].set_ylabel('Media de Fitness')
    ax[0].set_title('Media de Fitness por Generación')

    # # Configurar las marcas en el eje x para que comiencen en 1 y luego marquen cada 10
    # ax[0].set_xticks(np.concatenate([[1], np.arange(50, len(means) + 1, 50)]))

    # Mostrar cuadrícula en el gráfico
    ax[0].grid(True)

    # Texto de hiperparámetros en el subplot derecho
    hyperparameters = {
        'Condición de corte': f'{config['hyperparams']['termination']['amount']} gen',
        'Tiempo máximo': f'{config['time_limit']}s',
        'Selección 1': 'Elite 100%',
        'K': f"{config['hyperparams']['selection']['k']}",
        'Mutación': 'Sinle Gene',
        'P(mutación)': '1%',
        'Cruza': 'One Point',
        'Reemplazo': 'FillAll',
        'Selección 1 de reemplazo': 'Elite 100%',
        'Puntos totales': f'{config['total_points']}',
        'Cantidad Runs': f'{config['runs']}',
        'Tamaño de poblacion': f'{config["population_size"]}',
        'Tipo': f'Guerrero',
    }

    # Alinear el texto a la izquierda dentro del subplot derecho
    hyperparams_text = "\n".join([f"{param}: {value}" for param, value in hyperparameters.items()])
    ax[1].text(0, 1, hyperparams_text, transform=ax[1].transAxes, fontsize=12, verticalalignment='top',
               horizontalalignment='left')

    # Quitar el eje de la columna del texto
    ax[1].axis('off')

    # Ajustar el layout para que no se superpongan
    plt.tight_layout()
    plt.show()


def plot_max_fitness_per_generation(max_fitness_history_runs: List[List[float]], config):
    np_matrix = np.array(max_fitness_history_runs)
    max_fitness_history_runs = np_matrix.T
    means = []
    std_devs = []
    for run in max_fitness_history_runs:
        means.append(np.mean(run))
        std_devs.append(np.std(run))
    print(max_fitness_history_runs)
    print(means)
    print(std_devs)
    x = np.arange(1, len(means) + 1)

    # Crear figura y subplots (2 columnas: 70% gráfico, 30% texto)
    fig, ax = plt.subplots(figsize=(10, 6), nrows=1, ncols=2, gridspec_kw={'width_ratios': [3, 1]})

    # Gráfico de barras de error en el subplot izquierdo
    ax[0].errorbar(x, means, yerr=std_devs, fmt='o', capsize=5, linestyle='None', marker='o', color='b', ecolor='r')

    # Agregar etiquetas y título
    ax[0].set_xlabel('Generaciones')
    ax[0].set_ylabel('Maximo Fitness')
    ax[0].set_title('Maximo Fitness por Generación')

    # Configurar las marcas en el eje x para que comiencen en 1 y luego marquen cada 10
    ax[0].set_xticks(np.concatenate([[1], np.arange(10, len(means) + 1, 10)]))

    # Mostrar cuadrícula en el gráfico
    ax[0].grid(True)

    # Texto de hiperparámetros en el subplot derecho
    hyperparameters = {
        'Condición de corte': f'{config['hyperparams']['termination']['amount']} gen',
        'Tiempo máximo': f'{config['time_limit']}s',
        'Selección 1': 'Torneo Probabilistico(90%)',
        'Threshold': '0.6',
        'Selección 2': 'Ruleta (10%)',
        'K': f"{config['hyperparams']['selection']['k']}",
        'Mutación': 'Sinle Gene',
        'P(mutación)': '1%',
        'Cruza': 'One Point',
        'Reemplazo': 'FillAll',
        'Selección de reemplazo 1': 'Torneo Probabilistico(90%)',
        'Threshold reemplazo': '0.6',
        'Selección de reemplazo 2': 'Ruleta (10%)',
        'Puntos totales': f'{config['total_points']}',
        'Cantidad Runs': f'{config['runs']}',
        'Tamaño de poblacion': f'{config["population_size"]}',
        'Tipo': f'Arquero',
    }

    # Alinear el texto a la izquierda dentro del subplot derecho
    hyperparams_text = "\n".join([f"{param}: {value}" for param, value in hyperparameters.items()])
    ax[1].text(0, 1, hyperparams_text, transform=ax[1].transAxes, fontsize=12, verticalalignment='top',
               horizontalalignment='left')

    # Quitar el eje de la columna del texto
    ax[1].axis('off')

    # Ajustar el layout para que no se superpongan
    plt.tight_layout()
    plt.show()


def compute_means_and_stds(data: List[List[Tuple[float, float]]]):
    print(data)
    # Convertir la matriz de tuplas a un array de numpy
    np_matrix = np.array(data)

    # Separar las componentes de las tuplas
    first_elements = np_matrix[:, :, 0]
    second_elements = np_matrix[:, :, 1]

    # Transponer cada componente por separado
    means_matrix = first_elements.T
    std_devs_matrix = second_elements.T
    means = []
    std_devs = []
    for list_means, list_std_devs in zip(means_matrix, std_devs_matrix):
        means.append(np.mean(list_means))
        std_devs.append(np.sqrt(np.sum(np.square(list_std_devs)) / len(
            list_std_devs)))  # ddof=0 es para calcular la desviación estándar poblacional

    return means, std_devs
