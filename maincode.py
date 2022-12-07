from math import sqrt
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale
import numpy as np

path_data = Path(".") # si votre fichier de données n'est pas à l'endroit où vous exécutez ce code il faudra modifier le path
colmap = ["blue", "red", "olive", "green", "black"] # si vous avez plus de groupes il faudra rajouter des couleurs
places = ["Q", "E", "F", "M", "S", "m"] # m représente la MSH (pour éviter les confusions avec M)


# Cette fonction produit les représentations de proximité entre les graphes en se basant sur les distances
# attention : si l'ACP est déterministe, le t-SNE ne l'est pas ce qui signifie que les représentations peuvent
# différer si vous lancez le programme plusieurs fois (mais les mêmes proximités devraient être détectées à chaque fois)
def projections():
    df = pd.read_csv(path_data / "initial_data.csv", sep=";", encoding="utf-8") # fichier où se trouvent les données
    # attention j'ai dû modifier un peu la structure du fichier pour que ça fonctionne (j'ai tout mis en ligne)
    X = df.set_index("Unnamed: 0")
    Xs = minmax_scale(X.values) # là je normalise les valeurs entre les différents graphes
    print(df.info())

    pca = PCA()
    Xtr = pca.fit_transform(Xs) # c'est cette fonction qui fait la projection
    # les lignes qui suivent servent à générer l'affichage
    for k in range(2):
        plt.figure(figsize=(10, 10))
        plt.scatter(Xtr[:, 2 * k], Xtr[:, 2 * k + 1], c=[colmap[col // 6] for col in range(df.shape[0])]) # affichage des points
        for i, row in df.iterrows():
            plt.annotate(row["Unnamed: 0"], (Xtr[i, 2 * k] + 0.02, Xtr[i, 2 * k + 1] - 0.015)) # légendes des points
        loadings: np.ndarray[float] = pca.components_.T * np.sqrt(pca.explained_variance_)
        for j, feature in enumerate(X.columns): # affichage des axes projetés
            plt.gca().add_line(plt.Line2D((0, loadings[j, 2 * k] * 3), (0, loadings[j, 2 * k + 1] * 3), color="gray"))
            plt.annotate(feature, xy=(loadings[j, 2 * k] * 3, loadings[j, 2 * k + 1] * 3))
        plt.savefig(path_data / f"PCA_normalisée_axes{2 * k}-{2 * k + 1}.png") # endroit où est sauvegardé le résultat
    plt.close()

    # la même chose, mais sans normalisation
    pca = PCA()
    Xtr = pca.fit_transform(X.values)
    for k in range(2):
        plt.figure(figsize=(10, 10))
        plt.scatter(Xtr[:, 2 * k], Xtr[:, 2 * k + 1], c=[colmap[col // 6] for col in range(df.shape[0])])
        for i, row in df.iterrows():
            plt.annotate(row["Unnamed: 0"], (Xtr[i, 2 * k] + 0.02, Xtr[i, 2 * k + 1] - 0.015))
        loadings: np.ndarray[float] = pca.components_.T * np.sqrt(pca.explained_variance_)
        print(pca.explained_variance_ratio_)
        for j, feature in enumerate(X.columns):
            plt.gca().add_line(plt.Line2D((0, loadings[j, 2 * k] * 3), (0, loadings[j, 2 * k + 1] * 3), color="gray"))
            plt.annotate(feature, xy=(loadings[j, 2 * k] * 3, loadings[j, 2 * k + 1] * 3))
        plt.savefig(path_data / f"PCA_brut_axes{2 * k}-{2 * k + 1}.png")
    plt.close()

    # une projection alternative (t-SNE) qui préserve très bien les proximités, version non normalisée
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)
    Xtr = tsne.fit_transform(X.values)
    plt.figure(figsize=(10, 10))
    plt.scatter(Xtr[:, 0], Xtr[:, 1], c=[colmap[i // 6] for i in range(df.shape[0])])
    for i, row in df.iterrows():
        plt.annotate(row["Unnamed: 0"], (Xtr[i, 0] + 0.02, Xtr[i, 1] - 0.015))
    plt.savefig(path_data / f"TSNE_brut.png")

    # même chose, mais normalisé cette fois
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)
    Xtr = tsne.fit_transform(Xs)
    plt.figure(figsize=(10, 10))
    plt.scatter(Xtr[:, 0], Xtr[:, 1], c=[colmap[i // 6] for i in range(df.shape[0])])
    for i, row in df.iterrows():
        plt.annotate(row["Unnamed: 0"], (Xtr[i, 0] + 0.02, Xtr[i, 1] - 0.015))
    plt.savefig(path_data / f"TSNE_normalisé.png")

    plt.close()

# cette fonction effectue les calculs de centralité
def centralite():
    df = pd.read_csv(path_data / "initial_data.csv", sep=";", encoding="utf-8")
    X = df.set_index("Unnamed: 0")
    for col in X.columns:
        X.loc[:, col] = X.loc[:, col].apply(lambda z: 1 / z) # on prend comme poids l'inverse des distances
    X = X.rename({"QMSH": "Qm", "EMSH": "Em", "FMSH": "Fm", "MMSH": "Mm", "MSHS": "mS"}, axis=1) # j'ai remplacé "MSH" par "m" pour simplifier le code
    for col in "QEFMSm":
        X.loc[:, col] = X.loc[:, [c for c in X.columns if col in c]].sum(axis=1).multiply(100) # calcul de la centralité des points
    X.to_csv(path_data / "centralite.csv", sep=";", encoding="utf-8")


# je refais les mêmes projections que dans la fonction initiale, mais cette fois avec les données de centralité
def projections_centralites():
    df = pd.read_csv(path_data / "centralite.csv", sep=";", encoding="utf-8")
    X = df.set_index("Unnamed: 0")
    X = X.loc[:, places]
    Xs = minmax_scale(X.values)
    print(df.info())

    pca = PCA()
    Xtr = pca.fit_transform(Xs)
    print(pca.explained_variance_ratio_)
    for k in range(2):
        plt.figure(figsize=(10, 10))
        plt.scatter(Xtr[:, 2 * k], Xtr[:, 2 * k + 1], c=[colmap[col // 6] for col in range(df.shape[0])])
        for i, row in df.iterrows():
            plt.annotate(row["Unnamed: 0"], (Xtr[i, 2 * k] + 0.02, Xtr[i, 2 * k + 1] - 0.015))
        loadings: np.ndarray[float] = pca.components_.T * np.sqrt(pca.explained_variance_)
        for j, feature in enumerate(X.columns):
            plt.gca().add_line(plt.Line2D((0, loadings[j, 2 * k] * 3), (0, loadings[j, 2 * k + 1] * 3), color="gray"))
            plt.annotate(feature, xy=(loadings[j, 2 * k] * 3, loadings[j, 2 * k + 1] * 3))
        plt.savefig(path_data / f"centr_PCA_normalisée_axes{2 * k}-{2 * k + 1}.png")
    plt.close()

    pca = PCA()
    Xtr = pca.fit_transform(X.values)
    for k in range(2):
        plt.figure(figsize=(10, 10))
        plt.scatter(Xtr[:, 2 * k], Xtr[:, 2 * k + 1], c=[colmap[col // 6] for col in range(df.shape[0])])
        for i, row in df.iterrows():
            plt.annotate(row["Unnamed: 0"], (Xtr[i, 2 * k] + 0.02, Xtr[i, 2 * k + 1] - 0.015))
        loadings: np.ndarray[float] = pca.components_.T * np.sqrt(pca.explained_variance_)
        for j, feature in enumerate(X.columns):
            plt.gca().add_line(plt.Line2D((0, loadings[j, 2 * k] * 3), (0, loadings[j, 2 * k + 1] * 3), color="gray"))
            plt.annotate(feature, xy=(loadings[j, 2 * k] * 3, loadings[j, 2 * k + 1] * 3))
        plt.savefig(path_data / f"centr_PCA_brut_axes{2 * k}-{2 * k + 1}.png")
    plt.close()

    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)
    Xtr = tsne.fit_transform(X.values)
    plt.figure(figsize=(10, 10))
    plt.scatter(Xtr[:, 0], Xtr[:, 1], c=[colmap[i // 6] for i in range(df.shape[0])])
    for i, row in df.iterrows():
        plt.annotate(row["Unnamed: 0"], (Xtr[i, 0] + 0.02, Xtr[i, 1] - 0.015))
    plt.savefig(path_data / f"centr_TSNE_brut.png")

    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)
    Xtr = tsne.fit_transform(Xs)
    plt.figure(figsize=(10, 10))
    plt.scatter(Xtr[:, 0], Xtr[:, 1], c=[colmap[i // 6] for i in range(df.shape[0])])
    for i, row in df.iterrows():
        plt.annotate(row["Unnamed: 0"], (Xtr[i, 0] + 0.02, Xtr[i, 1] - 0.015))
    plt.savefig(path_data / f"centr_TSNE_normalisé.png")

    plt.close()

# cette fonction sert juste à calculer des indicateurs de centralité moyens par groupe. il faudra la modifier un peu si
# les groupes ne sont plus de la même taille que ceux considérés jusque ici
def indicateurs():
    df = pd.read_csv(path_data / "centralite.csv", sep=";", encoding="utf-8")
    X = df.set_index("Unnamed: 0")
    X = X.loc[:, places]
    Xs = pd.DataFrame(minmax_scale(X.values))
    output = "& Q & E & F & M & S MSH\\\\\n"
    for i in range(4):
        print("groupe", i + 1)
        Xloc = Xs.iloc[range(6 * i, 6 * (i + 1)), :] # je ne garde que les éléments du groupe (il y en a exactement 6)
        moy = Xloc.mean(axis=0)
        output += "groupe" + str(i) + " & " + " & ".join([str(u) for u in moy.to_list()]) + "\\\\\n"
    print(output)


# cette fonction calcule les aires des triangles puis fait une ACP
def calculs_aires():
    df = pd.read_csv(path_data / "initial_data.csv", sep=";", encoding="utf-8")
    X = df.set_index("Unnamed: 0")
    X = X.rename({"QMSH": "Qm", "EMSH": "Em", "FMSH": "Fm", "MMSH": "Mm", "MSHS": "mS"}, axis=1)
    out: pd.DataFrame = X.copy()
    # là on va générer tous les triangles et appliquer la formule de Héron
    for i1, place1 in enumerate(places):
        for i2, place2 in enumerate(places):
            if i2 > i1:
                for i3, place3 in enumerate(places):
                    if i3 > i2:
                        triangle = list(set([c for c in place1+place2+place3]))
                        print(triangle)
                        edges = [e for e in X.columns if e[0] in triangle and e[1] in triangle]
                        out.loc[:, "-".join(triangle)] = X.loc[:, edges].apply(heron, axis=1)
    out = out.drop(X.columns, axis=1) # on supprime les colonnes qui ne nous servent plus avant de faire l'ACP
    Xs = minmax_scale(out.values)
    pca = PCA()
    Xtr = pca.fit_transform(Xs)
    print(pca.explained_variance_ratio_)
    for k in range(2):
        plt.figure(figsize=(10, 10))
        plt.scatter(Xtr[:, 2 * k], Xtr[:, 2 * k + 1], c=[colmap[col // 6] for col in range(df.shape[0])])
        for i, row in df.iterrows():
            plt.annotate(row["Unnamed: 0"], (Xtr[i, 2 * k] + 0.02, Xtr[i, 2 * k + 1] - 0.015))
        loadings: np.ndarray[float] = pca.components_.T * np.sqrt(pca.explained_variance_)
        for j, feature in enumerate(out.columns):
            plt.gca().add_line(plt.Line2D((0, loadings[j, 2 * k] * 3), (0, loadings[j, 2 * k + 1] * 3), color="gray"))
            plt.annotate(feature, xy=(loadings[j, 2 * k] * 3, loadings[j, 2 * k + 1] * 3))
        plt.savefig(path_data / f"PCA_aires_axes{2 * k}-{2 * k + 1}.png")
    plt.close()


# aire d'un triangle à partir des longueurs de ses côtés
# pour pallier les problèmes liés aux erreurs de mesures,
# je fixe à 0 les cas où l'inégalité triangulaire n'est pas vérifiée
def heron(edges: pd.Series) -> int:
    p = edges.sum() / 2
    if p*(p-edges[0])*(p-edges[1])*(p-edges[2]) < 0:
        return 0
    else:
        return int(sqrt(p*(p-edges[0])*(p-edges[1])*(p-edges[2])))


# le coeur du programme, qui appelle successivement toutes les fonctions.
if __name__ == "__main__":
    projections()
    centralite()
    projections_centralites()
    indicateurs()
    calculs_aires()
