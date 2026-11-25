# comparar os 3 metodos no final

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import ajuda

os.makedirs("figuras", exist_ok=True)
os.makedirs("resultados", exist_ok=True)


def avaliar_tudo(nome, X, y):
    X_tr, X_te, y_tr, y_te = ajuda.split_padrao(X, y)
    linhas = []

    print(nome, "svm")
    g1 = GridSearchCV(
        Pipeline([("scaler", StandardScaler()), ("m", SVC())]),
        [
            {"m__kernel": ["linear"], "m__C": [0.1, 1, 10]},
            {"m__kernel": ["rbf"], "m__C": [0.1, 1, 10], "m__gamma": ["scale", 0.1]},
        ],
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1,
    )
    g1.fit(X_tr, y_tr)
    m1 = ajuda.metricas(y_te, g1.predict(X_te), "SVM")
    m1["dataset"] = nome
    linhas.append(m1)

    print(nome, "logistica")
    g2 = GridSearchCV(
        Pipeline([("scaler", StandardScaler()), ("m", LogisticRegression(max_iter=2000))]),
        {"m__C": [0.1, 1, 10]},
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1,
    )
    g2.fit(X_tr, y_tr)
    m2 = ajuda.metricas(y_te, g2.predict(X_te), "Logistica")
    m2["dataset"] = nome
    linhas.append(m2)

    print(nome, "knn")
    g3 = GridSearchCV(
        Pipeline([("scaler", StandardScaler()), ("m", KNeighborsClassifier())]),
        {"m__n_neighbors": [3, 5, 7, 11, 15], "m__weights": ["uniform", "distance"]},
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1,
    )
    g3.fit(X_tr, y_tr)
    m3 = ajuda.metricas(y_te, g3.predict(X_te), "kNN")
    m3["dataset"] = nome
    linhas.append(m3)

    return linhas


todas = []
X_cdc, y_cdc, df_cdc = ajuda.carregar_cdc()
todas += avaliar_tudo("cdc", X_cdc, y_cdc)

X_p, y_p, df_p, le = ajuda.carregar_pumpkin()
todas += avaliar_tudo("pumpkin", X_p, y_p)

df = pd.DataFrame([{k: v for k, v in r.items() if k != "cm"} for r in todas])
df.to_csv("resultados/comparacao.csv", index=False)
print(df)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, ds in zip(axes, ["cdc", "pumpkin"]):
    ped = df[df["dataset"] == ds]
    bars = ax.bar(ped["modelo"], ped["f1"], edgecolor="black")
    for b, h in zip(bars, ["///", "xx", ".."]):
        b.set_hatch(h)
    for i, v in enumerate(ped["f1"].values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_title(ds)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
plt.tight_layout()
plt.savefig("figuras/painel_comparacao.png", dpi=120)
plt.close()

for ds in ["cdc", "pumpkin"]:
    ped = df[df["dataset"] == ds][["modelo", "acc", "prec", "rec", "f1"]].reset_index(drop=True)
    ajuda.plot_stream_metricas(ped, f"figuras/stream_{ds}.png")

print("melhor por dataset:")
for ds in ["cdc", "pumpkin"]:
    ped = df[df["dataset"] == ds].sort_values("f1", ascending=False)
    print(ds, ped.iloc[0]["modelo"], round(ped.iloc[0]["f1"], 4), "diff:", round(ped.iloc[0]["f1"] - ped.iloc[1]["f1"], 4))
print("fim")
