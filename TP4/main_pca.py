import pandas as pd

from TP4.src.PCA.PCA_plotting import plot_nonstandard_data, plot_standard_data, plot_biplot, plot_pc1, \
    plot_variance_ratio, plot_components


def main():
    df = pd.read_csv("./inputs/europe.csv")
    print('df: ', df)
    plot_nonstandard_data(df)
    plot_standard_data(df)
    plot_biplot(df)
    plot_pc1(df)
    plot_variance_ratio(df)
    plot_components(df)


if __name__ == "__main__":
    main()
