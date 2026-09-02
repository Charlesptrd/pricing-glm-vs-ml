import matplotlib.pyplot as plt
import pandas as pd

def freq_par_groupe(df, col, bins=None):
    d = df.copy()

    if bins is not None:
        d[col] = pd.cut(d[col], bins=bins)

    freq_groupe = d.groupby(col, observed=True).agg(
        exposition=("Exposure", "sum"),
        sinistres=("ClaimNb", "sum"),
    )
    print(freq_groupe.columns)
    freq_groupe["frequence"] = freq_groupe["sinistres"] / freq_groupe["exposition"]

    return freq_groupe


def plot_freq_par_groupe(df, col, bins=None, ax=None):
    g = freq_par_groupe(df, col, bins)

    if ax is None:                       # usage isolé
        fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(g))
    ax.bar(x, g["exposition"], color="lightsteelblue")
    ax.set_ylabel("Exposition")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(i) for i in g.index], rotation=45, ha="right", fontsize=8)

    ax2 = ax.twinx()
    ax2.plot(x, g["frequence"], "o-", color="crimson", ms=4)
    ax2.set_ylabel("Fréquence")
    ax2.axhline(df["ClaimNb"].sum() / df["Exposure"].sum(),
                color="gray", ls="--", lw=1)

    ax.set_title(col)
    return g