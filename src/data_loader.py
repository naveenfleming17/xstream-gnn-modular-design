import pandas as pd
import json

def load_csv(file_path):
    return pd.read_csv(file_path)

def clean_radiation_column(df):
    col = 'yearly_overall_radiation_wh/sqm'
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace('"', '').str.replace(',', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
