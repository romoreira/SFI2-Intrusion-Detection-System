# Importar a biblioteca torch
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Definir o modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.dropout(x)  # Aplicar o dropout após a ativação
        x = self.output_layer(x)
        # Aplicar a função Softmax na camada de saída
        x = nn.functional.softmax(x, dim=1)
        return x


# Inicializar o modelo
model1 = LSTMModel(input_size=78, hidden_size=16, num_layers=5, output_size=2)

# Carregar o estado do modelo
state_dict1 = torch.load("../results/cic-unb-models/server_model_aggregated.pth")

# Carregar o estado no modelo
model1.load_state_dict(state_dict1)

# Acessar os parâmetros do modelo
params_server = model1.state_dict()

# Converter os parâmetros em arrays numpy
for key, value in params_server.items():
    params_server[key] = value.numpy()

#print(model1)

#---------------------------------------------------------------------------------------

# Percorrer os modelos dos clientes de 1 a 7
for client_id in range(1, 8):
    # Inicializar o modelo do cliente
    model_client = []
    model_client = LSTMModel(input_size=78, hidden_size=16, num_layers=5, output_size=2)

    # Carregar o estado do modelo do cliente
    state_dict_client = torch.load(f"../results/cic-unb-models/{client_id}_cic_unb_client_model_aggregated.pth")

    # Carregar o estado no modelo do cliente
    model_client.load_state_dict(state_dict_client)

    params_server = parameters_to_vector(model1.parameters())
    params_client = parameters_to_vector(model_client.parameters())

    cos = nn.CosineSimilarity(dim=0)
    div = cos(params_server, params_client)

    print(f"Similaridade de cosseno entre o modelo do servidor e o modelo do cliente {client_id}: {div}")