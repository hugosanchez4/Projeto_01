# regressao logistica

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import ajuda

os.makedirs("figuras", exist_ok=True)
os.makedirs("resultados", exist_ok=True)

resultados = []


def roda_log(nome_dados, X, y, labels, feats_fronteira=None, nomes_eixos=None):
    print("\n######## LOG -", nome_dados, "########")
    X_tr, X_te, y_tr, y_te = ajuda.split_padrao(X, y)

    pipe = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]
    )
    grid = GridSearchCV(pipe, {"clf__C": [0.1, 1, 10]}, cv=3, scoring="f1_weighted", n_jobs=-1)
    grid.fit(X_tr, y_tr)
    print("melhor:", grid.best_params_)

    pred = grid.predict(X_te)
    m = ajuda.metricas(y_te, pred, nome=f"LOG_{nome_dados}")
    resultados.append(m)

    ajuda.plot_matriz_hatch(m["cm"], labels, f"log {nome_dados}", f"figuras/log_cm_{nome_dados}.png")
    ajuda.plot_circulo_confusao(m["cm"], labels, f"log {nome_dados}", f"figuras/log_circ_{nome_dados}.png")

    if feats_fronteira is not None:
        X2 = X[feats_fronteira].values
        X2_tr, X2_te, y2_tr, y2_te = ajuda.split_padrao(X2, y)
        C = grid.best_params_["clf__C"]
        pipe2 = Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, C=C))]
        )
        pipe2.fit(X2_tr, y2_tr)
        X2_esc = pipe2.named_steps["scaler"].transform(X2)
        ajuda.plot_fronteira(
            pipe2.named_steps["clf"],
            X2_esc,
            y.values if hasattr(y, "values") else y,
            nomes_eixos,
            f"log {nome_dados}",
            f"figuras/log_fronteira_{nome_dados}.png",
        )

    return m


X_cdc, y_cdc, df_cdc = ajuda.carregar_cdc()
roda_log("cdc", X_cdc, y_cdc, ["0", "1"], ["BMI", "Age"], ["BMI", "Age"])

X_p, y_p, df_p, le = ajuda.carregar_pumpkin()
roda_log("pumpkin", X_p, y_p, list(le.classes_), ["Area", "Perimeter"], ["Area", "Perimeter"])

pd.DataFrame([{k: v for k, v in r.items() if k != "cm"} for r in resultados]).to_csv(
    "resultados/log_resumo.csv", index=False
)
print("ok logistica")
