from src.data_loader import load_csv, clean_radiation_column
from src.graph_builder import build_graph_from_csv
from src.features import log_transform
from src.predict import load_model, load_scaler, run_inference
from src.evaluate import build_results_dataframe

import os

DATA_PATH = "data/raw/"
MODEL_PATH = "models/complete_model.pth"
SCALER_PATH = "models/label_scaler.pkl"

target_columns = [
    'avg_wind_around_combination_m/s',
    'area_percentage_low_wind',
    'avg_radiation_walls_wh/sqm',
    'area_percentage_high_radiation_wh/sqm',
    'overall_radiation_walls_wh/sqm',
    'area_percentage_visibility>50%',
    'egress_distance_sum_shortest_paths_m',
    'egress_distance_num_paths>60m',
    'yearly_energy_consumption_kWh'
]

graphs = {}

# Load data
for file in os.listdir(DATA_PATH):
    if file.endswith(".csv"):
        df = load_csv(os.path.join(DATA_PATH, file))
        df = clean_radiation_column(df)

        # Dummy site row (you can extend later)
        site_row = df.iloc[0]

        g = build_graph_from_csv(df, site_row)
        g.ndata['feature'] = log_transform(g.ndata['feature'])

        graphs[file] = g

# Load model
model = load_model(MODEL_PATH)
scaler = load_scaler(SCALER_PATH)

# Predict
predictions = run_inference(list(graphs.values()), model, scaler)

# Evaluate
df_results = build_results_dataframe(graphs, predictions, target_columns)

print(df_results.head())
