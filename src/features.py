import pandas as pd
import sys
import numpy as np

BINS_DRIVAGE = [17, 19, 21, 24, 29, 39, 49, 59, 69, 90]
LABELS_DRIVAGE = ["18-19", "20-21", "22-24", "25-29", "30-39","40-49", "50-59", "60-69", "70+"]

BINS_BM = [49, 50, 60, 75, 100, 350]
LABELS_BM = ["50", "51-60", "61-75", "76-100", "100+"]

BINS_DENSITY = [-1, 50, 200, 1000, 5000, 30000]
LABELS_DENSITY = ["very low", "low", "medium", "high", "very high"]

VEHAGE_BINS = [-1, 0, 1, 4, 10, 20, 31]
VEHAGE_LABELS = ['0', '1', '2-4', '5-10', '11-20', '21+']

SEUIL_EXPOSITION = 5000

def preparer(df:pd.DataFrame):
    df = df.copy()
    df["age_grp"] = pd.cut(df["DrivAge"], bins=BINS_DRIVAGE, labels=LABELS_DRIVAGE)
    df["bm_grp"] = pd.cut(df["BonusMalus"], bins=BINS_BM, labels=LABELS_BM)
    df["log_Density"] = np.log(df["Density"])
    df["dens_grp"] = pd.cut(df["Density"], bins=BINS_DENSITY, labels=LABELS_DENSITY)
    df["vehage_grp"] = pd.cut(df["VehAge"], bins=VEHAGE_BINS, labels=VEHAGE_LABELS)
    #print(df["vehage_grp"].value_counts())

    for c in ["age_grp", "bm_grp", "dens_grp", "vehage_grp"]:
        assert df[c].notna().all(), f"valeurs hors bornes dans {c}"
        assert df[c].nunique() > 1, f"{c} n'a qu'une modalité"

    return df


def modalites_frequentes(train:pd.DataFrame, col, seuil=SEUIL_EXPOSITION):
    expo = train.groupby(col, observed=True)["Exposure"].sum()

    return expo[expo >= SEUIL_EXPOSITION].index.tolist()

def regrouper(df:pd.DataFrame, col, modalites, autre="Autres"):
    d = df.copy()
    d[col] = d[col].astype(str).where(d[col].isin(modalites), autre)
    return d



