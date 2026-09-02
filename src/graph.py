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


def plot_freq_par_groupe(df, col, bins=None):
    freq_groupe = freq_par_groupe(df, col, bins)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.bar(range(len(freq_groupe)), freq_groupe["exposition"], color="lightsteelblue")
    ax1.set_ylabel("Exposition (années-police)")
    ax1.set_xticks(range(len(freq_groupe)))
    ax1.set_xticklabels([str(i) for i in freq_groupe.index], rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(range(len(freq_groupe)), freq_groupe["frequence"], "o-", color="crimson")
    ax2.set_ylabel("Fréquence (sinistres / année-police)")
    ax2.axhline(
        df["ClaimNb"].sum() / df["Exposure"].sum(),
        color="gray", linestyle="--", lw=1,
    )

    plt.title(f"Fréquence de sinistres par {col}")
    plt.tight_layout()
    plt.show()
    return freq_groupe