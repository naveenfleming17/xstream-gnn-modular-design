import torch

def log_transform(features_tensor):
    return torch.sign(features_tensor) * torch.log1p(torch.abs(features_tensor))

def inverse_log_transform(tensor):
    return torch.sign(tensor) * (torch.exp(torch.abs(tensor)) - 1)

def append_meteorological(node_features_tensor, meteo_data):
    meteo_tensor = torch.tensor(meteo_data, dtype=torch.float32).view(1, -1)
    meteo_tensor = meteo_tensor.repeat(node_features_tensor.shape[0], 1)
    return torch.cat((node_features_tensor, meteo_tensor), dim=1)
