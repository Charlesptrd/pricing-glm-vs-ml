from sklearn.datasets import fetch_openml
import pandas as pd
import os

def load_data():
    os.makedirs("../data", exist_ok=True)

    if os.path.exists("../data/freq.parquet"):
        freq = pd.read_parquet("../data/freq.parquet")
        sev  = pd.read_parquet("../data/sev.parquet")
    else:
        freq = fetch_openml(data_id=41214, as_frame=True).frame
        sev  = fetch_openml(data_id=41215, as_frame=True).frame
        freq.to_parquet("../data/freq.parquet")
        sev.to_parquet("../data/sev.parquet")

    freq['Exposure'] = freq['Exposure'].clip(upper=1)
    freq['ClaimNb']  = freq['ClaimNb'].clip(upper=4)
    freq['VehAge']   = freq['VehAge'].clip(upper=30)
    freq['DrivAge']  = freq['DrivAge'].clip(upper=90)
    freq['VehGas']   = freq['VehGas'].str.strip("'")

    return freq, sev


    