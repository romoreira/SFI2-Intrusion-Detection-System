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
model1 = LSTMModel(input_size=78, hidden_size=16, num_layers=5, output_size=2).to(DEVICE)

# Carregar o estado do modelo
state_dict1 = torch.load("../results/cic-unb-models/cic_unb_server_model_aggregated.pth")

# Carregar o estado no modelo
model1.load_state_dict(state_dict1)

# Acessar os parâmetros do modelo
params1 = model1.state_dict()

# Converter os parâmetros em arrays numpy
for key, value in params1.items():
    params1[key] = value.numpy()

#print(model1)

#---------------------------------------------------------------------------------------

# Inicializar o modelo
model2 = LSTMModel(input_size=78, hidden_size=16, num_layers=5, output_size=2).to(DEVICE)

# Carregar o estado do modelo
state_dict2 = torch.load("../results/cic-unb-models/1_cic_unb_client_model_aggregated.pth")

# Carregar o estado no modelo
model1.load_state_dict(state_dict2)

# Acessar os parâmetros do modelo
params2 = model2.state_dict()

# Converter os parâmetros em arrays numpy
for key, value in params2.items():
    params2[key] = value.numpy()

#print(model2)

# Extrair parâmetros e converter para vetores
params1 = parameters_to_vector(model1.parameters())
params2 = parameters_to_vector(model2.parameters())

# Calcular o produto escalar
dot_product = torch.dot(params1, params2)

# Calcular as magnitudes
magnitude1 = params1.norm()
magnitude2 = params2.norm()

# Calcular a similaridade do cosseno
cosine_similarity = dot_product / (magnitude1 * magnitude2)

print(cosine_similarity)
