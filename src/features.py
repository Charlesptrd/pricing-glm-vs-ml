import pandas as pd
import sys
import numpy as np
import statsmodels.api as sm



BINS_DRIVAGE_FIN = [17, 19, 21, 24, 29, 39, 49, 59, 69, 90]
LABELS_DRIVAGE_FIN = ["18-19", "20-21", "22-24", "25-29", "30-39","40-49", "50-59", "60-69", "70+"]

BINS_DRIVAGE_GROSSIER   = [17, 19, 21, 24, 29, 39, 49, 59, 90]
LABELS_DRIVAGE_GROSSIER = ["18-19", "20-21", "22-24", "25-29", "30-39", "40-49", "50-59", "60+"]

BINS_BM = [49, 50, 60, 75, 100, 350]
LABELS_BM = ["50", "51-60", "61-75", "76-100", "100+"]

BINS_DENSITY = [-1, 50, 200, 1000, 5000, 30000]
LABELS_DENSITY = ["very low", "low", "medium", "high", "very high"]

VEHAGE_BINS = [-1, 0, 1, 4, 10, 20, 31]
VEHAGE_LABELS = ['0', '1', '2-4', '5-10', '11-20', '21+']



SEUIL_EXPOSITION = 5000


COLONNES_MODELE = ["age_grp", "bm_grp", "vehage_grp", "dens_grp", "VehGas", "VehPower", "VehBrand", "Region"]
REFERENCES = {"age_grp": "30-39", "bm_grp": "50", "vehage_grp": "5-10", "dens_grp": "medium", "VehGas": "Regular", "VehPower": "6", "VehBrand": "B12", "Region":   "R24"}

# Formules obtenues à l'issue des tests sur le modèle de fréquence dans notebooks/02_glm_frequence.ipynb
FORMULE_FREQUENCE_FINALE = ("ClaimNb ~ C(age_grp2, Treatment('30-39')) + C(bm_grp, Treatment('50'))"
        " + C(dens_grp, Treatment('medium')) + C(vehage_grp, Treatment('5-10'))"
        " + C(VehGas, Treatment('Regular')) + C(vehpower_grp_2, Treatment('5-6'))"
        " + C(VehBrand, Treatment('B12')) + C(Region, Treatment('R24'))" 
         " + age40p:dense")

FUSION_AGE = {"50-59": "50+", "60+": "50+"}
GROUPES_POWER = {"4": "4", "5": "5-6", "6": "5-6", "7": "7", "8": "8", "9": "9-10", "10": "9-10", "11": "11", "12+": "12+"}
AGES_40P = ["40-49", "50-59", "60+"]
DENS_ELEVEE = ["high", "very high"]


def preparer(df:pd.DataFrame, decoupage):
    df = df.copy()
    if decoupage=="fin":
        df["age_grp"] = pd.cut(df["DrivAge"], bins=BINS_DRIVAGE_FIN, labels=LABELS_DRIVAGE_FIN)
    if decoupage=="grossier":
        df["age_grp"] = pd.cut(df["DrivAge"], bins=BINS_DRIVAGE_GROSSIER, labels=LABELS_DRIVAGE_GROSSIER)
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
    #Attention on change ici des catégorielles en chaine de caractères.
    d[col] = d[col].astype(str).where(d[col].isin(modalites), autre)
    return d

def appliquer_decisions_phase1(df:pd.DataFrame):
    
    d = df.copy()
    d["age_grp2"] = d["age_grp"].astype(str).replace(FUSION_AGE)
    d["vehpower_grp_2"] = d["VehPower"].astype(str).replace(GROUPES_POWER)
    d["age40p"] = d["age_grp"].astype(str).isin(AGES_40P).astype(int)
    d["dense"] = d["dens_grp"].astype(str).isin(DENS_ELEVEE).astype(int)

    return d


