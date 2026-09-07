from sklearn.datasets import fetch_openml
import pandas as pd
import os
from pathlib import Path
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_data():

    """Charge les tables freMTPL2 et applique les corrections d'intégrité.

    Les données brutes sont mises en cache au format parquet ; les corrections
    sont appliquées à chaque appel, de sorte que modifier les règles ci-dessous
    ne nécessite pas de vider le cache.

    Corrections appliquées (cf. notebooks/01_exploration.ipynb) :
      - Exposure plafonnée à 1 (1 224 polices > 1 année-police, impossible)
      - ClaimNb plafonné à 4  (16 polices, rupture de la décroissance)
      - VehAge plafonné à 30  (valeurs sentinelles à 99-100)
      - DrivAge plafonné à 90 (valeur sentinelle à 99)
      - VehGas : retrait des apostrophes (artefact de conversion R)

    Returns
    -------
    (freq, sev) : tuple de DataFrame
    """

    DATA_DIR.mkdir(exist_ok=True)
    freq_path = DATA_DIR / "freq.parquet"
    sev_path = DATA_DIR / "sev.parquet"

    if freq_path.exists() and sev_path.exists():
        freq = pd.read_parquet(freq_path)
        sev  = pd.read_parquet(sev_path)
    else:
        freq = fetch_openml(data_id=41214, as_frame=True).frame
        sev  = fetch_openml(data_id=41215, as_frame=True).frame
        freq.to_parquet(freq_path)
        sev.to_parquet(sev_path)

    freq = freq.copy()
    freq['Exposure'] = freq['Exposure'].clip(upper=1)
    freq['ClaimNb']  = freq["ClaimNb"].clip(upper=4)
    freq['VehAge']   = freq['VehAge'].clip(upper=30)
    freq['DrivAge']  = freq['DrivAge'].clip(upper=90)
    freq['VehGas']   = freq['VehGas'].astype(str).str.strip("'")

    return freq, sev


    