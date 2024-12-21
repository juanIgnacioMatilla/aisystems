from matplotlib import pyplot as plt

from TP4.src.model.oja.oja_rule import oja_pc1
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from TP4.src.utils import standardize_inputs
import seaborn as sns


def plot_pc1(df, pc1_values):
    # Crear un nuevo DataFrame con los valores de PC1 y los países
    pc1_df = pd.DataFrame({
        'PC1': pc1_values,
        'Country': df['Country']
    })

    num_countries = pc1_df.shape[0]

    # Crear un gráfico de barras para PC1
    plt.figure(figsize=auto_resize_plot(num_countries))
    sns.barplot(x='Country', y='PC1', data=pc1_df)
    plt.title('PC1 por País')
    plt.xticks(rotation=90)  # Rotar etiquetas del eje x
    plt.ylabel('PC1')
    plt.tight_layout()
    plt.show()


def auto_resize_plot(num_items):
    # Ajusta el tamaño de la figura en función del número de países
    return (max(10, num_items / 2), 6)


def main():
    df = pd.read_csv("./inputs/europe.csv")
    df_without_country = df.drop(columns=['Country'])
    X = standardize_inputs(df_without_country)
    X = X.values
    # Normalizar los datos
    X_mean = np.mean(X, axis=0)
    X = (X - X_mean) / np.std(X, axis=0)
    # Inicializa PCA y ajusta el modelo a los datos
    pca = PCA()
    pca.fit(X)

    # Obtiene los autovectores (componentes principales)
    autovectores = pca.components_

    # El autovector asociado al mayor autovalor es el primero
    mayor_autovector = autovectores[0]

    print("Cargas de PC1 calculadas con PCA (skit-learn):")
    print(mayor_autovector)
    # Calcular la primera componente principal
    principal_component_weights = oja_pc1(X)
    print("Cargas de PC1 calculadas con regla de Oja:", principal_component_weights)

    # Calcular PC1 multiplicando los datos normalizados por los pesos
    PC1 = np.dot(X, principal_component_weights)
    plot_pc1(df, PC1)
    # Agregar PC1 al DataFrame original
    df['PC1'] = PC1

    # Iterar sobre el DataFrame para imprimir cada país con su respectivo PC1
    print("PC1 for each country:")
    for index, row in df.iterrows():
        print(f"{row['Country']}: {row['PC1']:.3f}")


if __name__ == "__main__":
    main()
