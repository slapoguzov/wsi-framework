import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def show_two_dimensions_plot(vectors: [int, float], cluster, title=""):
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(vectors)
    # plot

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = x_pca[:, 0]
    y = x_pca[:, 1]

    plt.title(title)
    ax.scatter(x, y, c=cluster)

    for i, txt in enumerate(x):
        ax.annotate(str(i), (x[i], y[i]))

    plt.show()

