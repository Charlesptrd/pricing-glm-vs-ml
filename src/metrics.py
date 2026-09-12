import numpy as np
import pandas as pd

def table_frequence(df, col):
    t = df.groupby(col, observed=True).agg(exposition=("Exposure", "sum"), sinistres=("ClaimNb", "sum"),)
    t["frequence"] = t["sinistres"] / t["exposition"]
    t["sigma"] = np.sqrt(t["sinistres"]) / t["exposition"]
    return t

def comparer_tranches(df, col):
    t = table_frequence(df, col)

    f, s = t["frequence"].to_numpy(), t["sigma"].to_numpy()
    noms = t.index.to_numpy()

    ecart = np.abs(np.diff(f))
    sigma_ecart = np.sqrt(s[:-1] ** 2 + s[1:] ** 2)

    res = pd.DataFrame({
        "paire": [f"{a} / {b}" for a, b in zip(noms[:-1], noms[1:])],
        "ecart": ecart,
        "sigma_ecart": sigma_ecart,
        "rapport": ecart / sigma_ecart,
    })

    return res