# funcoes que eu fui colocando aqui pra nao repetir codigo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def carregar_cdc(caminho="dados/cdc_diabetes_amostra.csv"):
    df = pd.read_csv(caminho)
    y = df["Diabetes_binary"].astype(int)
    X = df.drop(columns=["Diabetes_binary"])
    return X, y, df


def carregar_pumpkin(caminho="dados/pumpkin_seeds.csv"):
    df = pd.read_csv(caminho)
    le = LabelEncoder()
    y = le.fit_transform(df["SeedType"])
    X = df.drop(columns=["SeedType"])
    return X, y, df, le


def split_padrao(X, y, teste=0.25, seed=7):
    # mesma seed em tudo senao nao da pra comparar
    return train_test_split(X, y, test_size=teste, random_state=seed, stratify=y)


def escalar(X_tr, X_te):
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te), sc


def metricas(y_true, y_pred, nome="modelo"):
    # weighted por causa do desbalanceamento do cdc
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print("----", nome, "----")
    print("acuracia:", round(acc, 4))
    print("precisao:", round(prec, 4))
    print("revocacao:", round(rec, 4))
    print("f1:", round(f1, 4))
    print("matriz confusao:\n", cm)
    return {"modelo": nome, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm}


def plot_matriz_hatch(cm, labels, titulo, saida):
    # matriz com textura nas celulas
    fig, ax = plt.subplots(figsize=(5, 4))
    n = cm.shape[0]
    hatches = ["///", "xxx", "...", "\\\\\\"]
    cores = ["orange", "teal", "gold", "steelblue"]
    for i in range(n):
        for j in range(n):
            rect = plt.Rectangle(
                (j, n - 1 - i),
                1,
                1,
                facecolor=cores[(i * n + j) % len(cores)],
                edgecolor="black",
                hatch=hatches[(i * n + j) % len(hatches)],
                alpha=0.8,
            )
            ax.add_patch(rect)
            ax.text(
                j + 0.5,
                n - 1 - i + 0.5,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=12,
                color="black",
            )
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(list(reversed(labels)))
    ax.set_xlabel("predito")
    ax.set_ylabel("real")
    ax.set_title(titulo)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(saida, dpi=120)
    plt.close()
    print("salvei", saida)


def plot_fronteira(modelo, X2, y, nomes_eixos, titulo, saida):
    # so roda com 2 colunas
    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = modelo.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    for classe in np.unique(y):
        pts = X2[y == classe]
        ax.scatter(pts[:, 0], pts[:, 1], s=25, label=str(classe), edgecolors="k", linewidths=0.3)
    ax.set_xlabel(nomes_eixos[0])
    ax.set_ylabel(nomes_eixos[1])
    ax.set_title(titulo)
    ax.legend()
    plt.tight_layout()
    plt.savefig(saida, dpi=120)
    plt.close()
    print("salvei", saida)


def plot_barras_pattern(nomes, valores, titulo, saida, ylabel="valor"):
    fig, ax = plt.subplots(figsize=(7, 4))
    hatches = ["///", "xx", "..", "++"]
    cores = ["C0", "C1", "C2", "C3"]
    bars = ax.bar(nomes, valores, color=cores[: len(nomes)], edgecolor="black")
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    for i, v in enumerate(valores):
        ax.text(i, v + max(valores) * 0.02, str(v), ha="center", fontsize=9)
    ax.set_ylim(0, max(valores) * 1.2 if max(valores) > 0 else 1)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(saida, dpi=120)
    plt.close()
    print("salvei", saida)


def plot_stream_metricas(df_res, saida):
    # metricas empilhadas tipo stream
    metricas_cols = ["acc", "prec", "rec", "f1"]
    modelos = df_res["modelo"].tolist()
    vals = np.array(df_res[metricas_cols].values.T, dtype=float)
    cum = np.cumsum(vals, axis=0)
    mid = cum[-1] / 2.0
    lower = cum - vals - mid
    upper = cum - mid

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(modelos))
    for i, m in enumerate(metricas_cols):
        ax.fill_between(xs, lower[i], upper[i], label=m, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(modelos, rotation=15)
    ax.set_title("metricas empilhadas")
    ax.legend()
    plt.tight_layout()
    plt.savefig(saida, dpi=120)
    plt.close()
    print("salvei", saida)


def plot_circulo_confusao(cm, labels, titulo, saida):
    # grafico circular da matriz (pycirclize)
    try:
        from pycirclize import Circos
    except Exception as e:
        print("nao deu pra plotar o circulo:", e)
        return

    names = [str(x) for x in labels]
    matrix = pd.DataFrame(cm, index=names, columns=names)
    circos = Circos.initialize_from_matrix(
        matrix,
        space=3,
        r_lim=(80, 100),
        cmap="tab10",
        label_kws=dict(size=10),
        link_kws=dict(direction=1, ec="black", lw=0.4),
    )
    fig = circos.plotfig()
    fig.suptitle(titulo)
    fig.savefig(saida, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("salvei", saida)
