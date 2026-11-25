# olhar os dados antes de treinar

import os
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_point, geom_boxplot, theme_bw, labs, scale_color_manual

import ajuda

os.makedirs("figuras", exist_ok=True)

print("=== CDC ===")
X_cdc, y_cdc, df_cdc = ajuda.carregar_cdc()
print("linhas:", len(df_cdc))
print(df_cdc["Diabetes_binary"].value_counts())
print(df_cdc[["BMI", "Age", "MentHlth", "PhysHlth", "GenHlth", "Income"]].describe())

# se chutar sempre 0 ja acerto bastante
prop0 = (df_cdc["Diabetes_binary"] == 0).mean()
print("baseline chutando 0:", round(prop0, 3))

df_plot = df_cdc.copy()
df_plot["classe"] = df_plot["Diabetes_binary"].astype(str)
p = (
    ggplot(df_plot, aes(x="classe", y="BMI", fill="classe"))
    + geom_boxplot(alpha=0.7)
    + theme_bw()
    + labs(title="BMI por classe", x="classe", y="BMI")
)
p.save("figuras/cdc_bmi_boxplot.png", dpi=120, verbose=False)

fig, ax = plt.subplots(figsize=(6, 4.5))
for c, cor in [(0, "C0"), (1, "C1")]:
    pedaco = df_cdc[df_cdc["Diabetes_binary"] == c]
    ax.scatter(pedaco["Age"], pedaco["BMI"], s=10, alpha=0.4, c=cor, label=str(c))
ax.set_xlabel("Age")
ax.set_ylabel("BMI")
ax.set_title("Age x BMI")
ax.legend()
plt.tight_layout()
plt.savefig("figuras/cdc_age_bmi.png", dpi=120)
plt.close()

ajuda.plot_barras_pattern(
    ["sem diabetes", "com diabetes"],
    df_cdc["Diabetes_binary"].value_counts().sort_index().tolist(),
    "classes CDC",
    "figuras/cdc_classes_hatch.png",
    ylabel="n",
)

print("\n=== Pumpkin ===")
X_p, y_p, df_p, le = ajuda.carregar_pumpkin()
print(df_p["SeedType"].value_counts())
print(df_p[["Area", "Perimeter", "Compactness", "Roundness"]].describe())

p2 = (
    ggplot(df_p, aes(x="Area", y="Perimeter", color="SeedType"))
    + geom_point(alpha=0.5, size=1.5)
    + theme_bw()
    + labs(title="Area x Perimeter")
    + scale_color_manual(values=["blue", "red"])
)
p2.save("figuras/pumpkin_area_perimetro.png", dpi=120, verbose=False)

ajuda.plot_barras_pattern(
    list(le.classes_),
    df_p["SeedType"].value_counts().loc[le.classes_].tolist(),
    "classes pumpkin",
    "figuras/pumpkin_classes_hatch.png",
    ylabel="n",
)

print("fim")
