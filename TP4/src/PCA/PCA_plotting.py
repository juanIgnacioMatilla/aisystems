import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from TP4.src.utils import standardize_inputs


def auto_resize_plot(num_items, base_width=1, base_height=6, max_width=16, max_height=8):
    width = min(max_width, base_width * num_items)
    height = min(max_height, base_height)
    return width, height


def plot_nonstandard_data(df):
    df_without_country = df.drop(columns=['Country'])
    num_columns = df_without_country.shape[1]

    plt.figure(figsize=auto_resize_plot(num_columns))
    sns.boxplot(data=df_without_country)
    plt.xticks(rotation=45)
    plt.title("Boxplot de las características")
    plt.tight_layout()
    plt.show()


def plot_standard_data(df):
    df_without_country = df.drop(columns=['Country'])
    df_without_country = standardize_inputs(df_without_country)
    num_columns = df_without_country.shape[1]

    plt.figure(figsize=auto_resize_plot(num_columns))
    sns.boxplot(data=df_without_country)
    plt.xticks(rotation=45)
    plt.title("Boxplot de las características estandarizadas")
    plt.tight_layout()
    plt.show()


def plot_biplot(df):
    pca = PCA(n_components=2)
    df_without_country_scaled = standardize_inputs(df.drop(columns=['Country']).to_numpy())
    principal_components = pca.fit_transform(df_without_country_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
    final_df = pd.concat([df["Country"], pca_df], axis=1)
    final_df["Country_number"] = final_df.index + 1

    # 7. Crear una nueva columna que combine el número con el nombre del país
    final_df['Country_with_number'] = final_df['Country_number'].astype(str) + ': ' + final_df['Country']

    plt.figure(figsize=(16, 8))
    sns.scatterplot(x='PC1', y='PC2', data=final_df, hue='Country_with_number', palette='Set1', s=100)

    # Añadir las direcciones de las variables originales
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    for i, feature in enumerate(df.drop(columns=['Country']).columns):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='r', alpha=0.5, head_width=0.05)
        plt.text(loadings[i, 0] * 1.1, loadings[i, 1] * 1.1, feature, color='r', ha='center', va='center')

    # Añadir números de cada país al lado de sus respectivos puntos
    for i in range(final_df.shape[0]):
        plt.text(final_df['PC1'][i] + 0.02, final_df['PC2'][i] + 0.02, final_df['Country_number'][i], fontsize=9)

    # Etiquetas y título
    plt.title('Biplot de las dos primeras componentes principales')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc=(1.05, 0.01), title="Countries")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_components(df):
    df_without_country = df.drop(columns=['Country'])
    df_without_country = standardize_inputs(df_without_country)
    # Crear el modelo PCA
    pca = PCA(n_components=1)
    pca.fit(df_without_country)

    # Obtener las cargas del primer componente principal (PC1)
    cargas_pc1 = pca.components_[0]
    print(cargas_pc1)
    # Etiquetas de las variables originales (reemplaza con tus variables)
    features = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']
    for i,feature in enumerate(features):
        print(f'{feature}: {cargas_pc1[i]:.3f}')
    # Colores: verde para positivas, rojo para negativas
    colors = ['green' if c >= 0 else 'red' for c in cargas_pc1]

    # Crear el gráfico
    plt.figure(figsize=(10, 6))

    # Graficar las barras con los colores diferenciados
    plt.bar(features, cargas_pc1, color=colors, alpha=0.7)

    # Etiquetas del gráfico
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Cargas', fontsize=12)
    plt.title('Cargas de PC1 por feature', fontsize=14)


    # Rotar etiquetas en el eje X si es necesario
    plt.xticks(rotation=45, ha='right')

    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()


def plot_variance_ratio(df):
    # Eliminar la columna 'Country' para que no interfiera con el PCA
    df = df.drop(columns=['Country'])
    # Crear y entrenar el modelo PCA
    pca = PCA(n_components=df.shape[1])
    pca.fit(df)
    print(pca.components_)
    # Obtener las proporciones de la varianza explicada por cada componente
    var_ratios = pca.explained_variance_ratio_
    # Graficar las proporciones en escala logarítmica
    pcs = np.arange(1, len(var_ratios) + 1)  # Índices de los componentes

    plt.figure(figsize=(8, 6))
    plt.scatter(pcs, var_ratios, color='b')

    # Escala logarítmica en el eje Y
    plt.yscale('log')

    # Etiquetas y título
    plt.xlabel('Componente Principal')
    plt.ylabel('Proporción de Varianza (escala logarítmica)')
    plt.title('Proporción de Varianza por Componente Principal')

    # Establecer etiquetas personalizadas para el eje X
    etiquetas_x = [f'PC{i}' for i in range(1, len(var_ratios) + 1)]
    plt.xticks(pcs, etiquetas_x)

    # Mostrar cuadrícula
    plt.grid(True)

    # Mostrar el gráfico
    plt.show()


def plot_pc1(df):
    df_without_country = df.drop(columns=['Country'])
    df_without_country = standardize_inputs(df_without_country)
    pca = PCA(n_components=1)

    principal_components = pca.fit_transform(df_without_country)
    pc1_df = pd.DataFrame(data=principal_components, columns=["PC1"])
    pc1_df['Country'] = df['Country']

    num_countries = pc1_df.shape[0]

    # 7. Crear un gráfico de barras para PCA1
    plt.figure(figsize=auto_resize_plot(num_countries))
    sns.barplot(x='Country', y='PC1', data=pc1_df)
    plt.title('PC1 por País')
    plt.xticks(rotation=90)  # Rotar etiquetas del eje x
    plt.ylabel('PC1')
    plt.tight_layout()
    plt.show()


