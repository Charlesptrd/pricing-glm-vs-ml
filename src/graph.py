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
