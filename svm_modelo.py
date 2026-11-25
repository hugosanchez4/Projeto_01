# svm linear e rbf

import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import ajuda

os.makedirs("figuras", exist_ok=True)
os.makedirs("resultados", exist_ok=True)

resultados = []


def roda_svm(nome_dados, X, y, labels, feats_fronteira=None, nomes_eixos=None):
    print("\n######## SVM -", nome_dados, "########")
    X_tr, X_te, y_tr, y_te = ajuda.split_padrao(X, y)

    pipe = Pipeline([("scaler", StandardScaler()), ("svm", SVC())])
    param = [
        {"svm__kernel": ["linear"], "svm__C": [0.1, 1, 10]},
        {"svm__kernel": ["rbf"], "svm__C": [0.1, 1, 10], "svm__gamma": ["scale", 0.1, 0.01]},
    ]

    # cv so no treino
    grid = GridSearchCV(pipe, param, cv=3, scoring="f1_weighted", n_jobs=-1)
    print("treinando...")
    grid.fit(X_tr, y_tr)
    print("melhor:", grid.best_params_, "cv:", round(grid.best_score_, 4))

    pred = grid.predict(X_te)
    m = ajuda.metricas(y_te, pred, nome=f"SVM_{nome_dados}")
    resultados.append(m)

    ajuda.plot_matriz_hatch(m["cm"], labels, f"SVM {nome_dados}", f"figuras/svm_cm_{nome_dados}.png")
    ajuda.plot_circulo_confusao(m["cm"], labels, f"SVM {nome_dados}", f"figuras/svm_circ_{nome_dados}.png")

    # fronteira com 2 features
    if feats_fronteira is not None:
        X2 = X[feats_fronteira].values
        X2_tr, X2_te, y2_tr, y2_te = ajuda.split_padrao(X2, y)
        melhor_kernel = grid.best_params_["svm__kernel"]
        C = grid.best_params_["svm__C"]
        gamma = grid.best_params_.get("svm__gamma", "scale")
        pipe2 = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel=melhor_kernel, C=C, gamma=gamma)),
            ]
        )
        pipe2.fit(X2_tr, y2_tr)
        X2_esc = pipe2.named_steps["scaler"].transform(X2)
        ajuda.plot_fronteira(
            pipe2.named_steps["svm"],
            X2_esc,
            y.values if hasattr(y, "values") else y,
            nomes_eixos,
            f"SVM ({melhor_kernel}) {nome_dados}",
            f"figuras/svm_fronteira_{nome_dados}.png",
        )

    return m


X_cdc, y_cdc, df_cdc = ajuda.carregar_cdc()
roda_svm("cdc", X_cdc, y_cdc, ["0", "1"], ["BMI", "Age"], ["BMI", "Age"])

X_p, y_p, df_p, le = ajuda.carregar_pumpkin()
roda_svm("pumpkin", X_p, y_p, list(le.classes_), ["Area", "Perimeter"], ["Area", "Perimeter"])

pd.DataFrame([{k: v for k, v in r.items() if k != "cm"} for r in resultados]).to_csv(
    "resultados/svm_resumo.csv", index=False
)
print("ok svm")
