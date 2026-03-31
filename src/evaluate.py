import pandas as pd
import torch
from features import inverse_log_transform

def build_results_dataframe(graphs, predictions, target_columns):
    results = []

    for key, pred in zip(graphs.keys(), predictions):
        g = graphs[key]

        meteo = g.ndata['feature'][0, -7:]
        meteo = inverse_log_transform(meteo).numpy().tolist()

        results.append([key] + meteo + pred.tolist())

    columns = ['id',
        'prevailing_wind', 'avg_wind_m/s', 'yearly_overall_sun_h',
        'yearly_overall_radiation_wh/sqm', 'yearly_avg_temp',
        'min_temp', 'max_temp'
    ] + target_columns

    return pd.DataFrame(results, columns=columns)
