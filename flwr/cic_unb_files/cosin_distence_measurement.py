# Importar a biblioteca torch
import torch

# Carregar o modelo
model = torch.load("../results/cic-unb-models/cic_unb_server_model_aggregated.pth")

# Acessar os parâmetros do modelo
params = model.state_dict()

# Converter os parâmetros em arrays numpy
for key, value in params.items():
    params[key] = value.numpy()

print(model)