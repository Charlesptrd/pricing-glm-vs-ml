import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def freq_par_groupe(data_frame, colonnes, bins=None):
    d = data_frame.copy()

    if bins is not None:
        d[colonnes] = pd.cut(d[colonnes], bins=bins)

    freq_groupe = d.groupby(colonnes, observed=True).agg(exposition=("Exposure", "sum"), sinistres=("ClaimNb", "sum"))
    freq_groupe["frequence"] = freq_groupe["sinistres"] / freq_groupe["exposition"]

    return freq_groupe




def plot_freq_par_groupe(data_frame, colonnes, bins=None, ax=None, errorbar=False):
    graphique = freq_par_groupe(data_frame, colonnes, bins)

    if ax is None:          
        fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(graphique))
    ax.bar(x, graphique["exposition"], color="lightsteelblue")
    ax.set_ylabel("Exposition")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(i) for i in graphique.index], rotation=45, ha="right", fontsize=8)



    ax2 = ax.twinx()
    if errorbar:
        graphique["erreur"] = graphique["frequence"] / np.sqrt(graphique["sinistres"])
        ax2.errorbar(x, graphique["frequence"], yerr=graphique["erreur"], fmt="o", color="crimson", ms=4, capsize=3)

    ax2.plot(x, graphique["frequence"], "o-", color="crimson", ms=4)
    ax2.set_ylabel("Fréquence")
    ax2.axhline(data_frame["ClaimNb"].sum() / data_frame["Exposure"].sum(),color="gray", ls="--", lw=1)

    ax.set_title(colonnes)
    return graphique

def plot_facteurs(modele, prefixe, ref="référence", ax=None):
    
    # Je suis passé par le paramètre regex et non prefix car sinon cela prenait aussi `vehage` pour la modalité `age`
    params = modele.params.filter(regex= "^" + prefixe + "_")

    noms = [c.replace(prefixe + "_", "") for c in params.index]
    noms = [ref] + noms
    facteurs = np.concatenate([[1.0], np.exp(params.to_numpy())])

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    x = range(len(noms))
    couleurs = ["gray"] + ["steelblue"] * (len(noms) - 1)
    ax.bar(x, facteurs, color=couleurs)
    ax.axhline(1.0, color="crimson", ls="--", lw=1)

    ax.set_xticks(list(x))
    ax.set_xticklabels(noms, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Facteur multiplicatif")
    ax.set_title(prefixe)
    return

def plot_exposition_par_modalite(df:pd.DataFrame, colonne):
    df_agg = df.groupby(colonne)["Exposure"].sum().reset_index()
    df_agg = df_agg.sort_values(by=colonne)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df_agg[colonne].astype(str), df_agg["Exposure"],color="gray")

    ax.set_title(f"Exposition totale par {colonne}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def table_severite(df, col, montant="montant_ecrete", forfaits=None):
    t = df.groupby(col, observed=True)[montant].agg(sinistres="size", cout_moyen="mean", ecart_type="std")
    t["sigma"] = t["ecart_type"] / np.sqrt(t["sinistres"])
    t["erreur_relative"] = t["sigma"] / t["cout_moyen"]
    t["cout_brut"] = df.groupby(col, observed=True)["ClaimAmount"].mean()

    if forfaits is not None:
        est_forfait = df["ClaimAmount"].round(2).isin(forfaits)
        t["part_forfaits"] = est_forfait.groupby(df[col], observed=True).mean()

    return t


def plot_sev_par_groupe(df, col, montant="montant_ecrete", ax=None):
    t = table_severite(df, col, montant)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(t))
    ax.bar(x, t["sinistres"], color="lightsteelblue")
    ax.set_ylabel("Nombre de sinistres")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(i) for i in t.index], rotation=45, ha="right", fontsize=8)

    ax2 = ax.twinx()
    ax2.errorbar(x, t["cout_moyen"], yerr=t["sigma"], fmt="o-", color="crimson", ms=4, capsize=3)
    ax2.axhline(df[montant].mean(), color="gray", ls="--", lw=1)
    ax2.set_ylabel("Coût moyen écrêté (€)")

    ax.set_title(col)
    return t

