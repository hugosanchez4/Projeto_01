# rodar esse primeiro pra baixar os csv

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import arff
import urllib.request

os.makedirs("dados", exist_ok=True)

# CDC (openml)
url_cdc = "https://www.openml.org/data/v1/download/22124404/cdc_diabetes.arff"
arff_cdc = "dados/cdc_tmp.arff"
print("baixando CDC...")
urllib.request.urlretrieve(url_cdc, arff_cdc)

data, meta = arff.loadarff(arff_cdc)
df_cdc = pd.DataFrame(data)
for c in df_cdc.columns:
    if df_cdc[c].dtype == object:
        df_cdc[c] = df_cdc[c].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

# o arquivo original eh enorme, peguei 8000 linhas
# seed=7 pra eu conseguir repetir
_, amostra = train_test_split(
    df_cdc, test_size=8000, stratify=df_cdc["Diabetes_binary"], random_state=7
)
amostra.to_csv("dados/cdc_diabetes_amostra.csv", index=False)
os.remove(arff_cdc)
print("CDC amostra:", amostra.shape)
print(amostra["Diabetes_binary"].value_counts())

# pumpkin seeds (mesmo dataset do kaggle, baixei pelo openml)
url_pump = "https://openml.org/data/v1/download/22125262/Pumpkin_Seeds.arff"
arff_pump = "dados/pump_tmp.arff"
print("baixando pumpkin seeds...")
urllib.request.urlretrieve(url_pump, arff_pump)

data2, meta2 = arff.loadarff(arff_pump)
df_p = pd.DataFrame(data2)
for c in df_p.columns:
    if df_p[c].dtype == object:
        df_p[c] = df_p[c].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

df_p.to_csv("dados/pumpkin_seeds.csv", index=False)
os.remove(arff_pump)
print("pumpkin:", df_p.shape)
print(df_p["SeedType"].value_counts())
print("pronto")
