import torch
import dgl
import joblib

def load_model(model_path):
    return torch.load(model_path, map_location=torch.device('cpu'))

def load_scaler(scaler_path):
    return joblib.load(scaler_path)

def run_inference(graphs, model, scaler):
    batched = dgl.batch([dgl.add_self_loop(g) for g in graphs])

    node_features = batched.ndata['feature']
    edge_features = batched.edata['feature']
    edge_weights = batched.edata['weight']

    model.eval()
    with torch.no_grad():
        preds = model(batched, node_features, edge_features, edge_weights)
        preds = scaler.inverse_transform(preds.numpy())

    return preds
