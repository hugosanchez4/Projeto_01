# knn - o "treino" so guarda os pontos

import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import ajuda

os.makedirs("figuras", exist_ok=True)
os.makedirs("resultados", exist_ok=True)

resultados = []


def roda_knn(nome_dados, X, y, labels, feats_fronteira=None, nomes_eixos=None):
    print("\n######## kNN -", nome_dados, "########")
    X_tr, X_te, y_tr, y_te = ajuda.split_padrao(X, y)

    pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
    param = {
        "knn__n_neighbors": [3, 5, 7, 11, 15],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    }
    grid = GridSearchCV(pipe, param, cv=3, scoring="f1_weighted", n_jobs=-1)
    grid.fit(X_tr, y_tr)
    print("melhor:", grid.best_params_, "cv:", round(grid.best_score_, 4))

    # treino e teste
    print("treino:")
    ajuda.metricas(y_tr, grid.predict(X_tr), nome=f"kNN_tr_{nome_dados}")
    print("teste:")
    m = ajuda.metricas(y_te, grid.predict(X_te), nome=f"kNN_{nome_dados}")
    resultados.append(m)

    ajuda.plot_matriz_hatch(m["cm"], labels, f"knn {nome_dados}", f"figuras/knn_cm_{nome_dados}.png")
    ajuda.plot_circulo_confusao(m["cm"], labels, f"knn {nome_dados}", f"figuras/knn_circ_{nome_dados}.png")

    if feats_fronteira is not None:
        X2 = X[feats_fronteira].values
        X2_tr, X2_te, y2_tr, y2_te = ajuda.split_padrao(X2, y)
        k = grid.best_params_["knn__n_neighbors"]
        w = grid.best_params_["knn__weights"]
        p = grid.best_params_["knn__p"]
        pipe2 = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=k, weights=w, p=p)),
            ]
        )
        pipe2.fit(X2_tr, y2_tr)
        X2_esc = pipe2.named_steps["scaler"].transform(X2)
        ajuda.plot_fronteira(
            pipe2.named_steps["knn"],
            X2_esc,
            y.values if hasattr(y, "values") else y,
            nomes_eixos,
            f"knn k={k} {nome_dados}",
            f"figuras/knn_fronteira_{nome_dados}.png",
        )

    return m


X_cdc, y_cdc, df_cdc = ajuda.carregar_cdc()
roda_knn("cdc", X_cdc, y_cdc, ["0", "1"], ["BMI", "Age"], ["BMI", "Age"])

X_p, y_p, df_p, le = ajuda.carregar_pumpkin()
roda_knn("pumpkin", X_p, y_p, list(le.classes_), ["Area", "Perimeter"], ["Area", "Perimeter"])

pd.DataFrame([{k: v for k, v in r.items() if k != "cm"} for r in resultados]).to_csv(
    "resultados/knn_resumo.csv", index=False
)
print("ok knn")
