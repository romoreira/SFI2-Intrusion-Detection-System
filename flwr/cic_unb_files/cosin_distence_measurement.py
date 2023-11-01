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
state_dict1 = torch.load("../results/cic-unb-models/cic_unb_server_model_aggregated.pth")

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



    # Calcular o produto escalar
    dot_product = torch.dot(params_server, params_client)

    # Calcular as magnitudes
    magnitude_server = params_server.norm()
    magnitude_client = params_client.norm()

    # Calcular a similaridade do cosseno
    cosine_similarity = dot_product / (magnitude_server * magnitude_client)

    print(f"Similaridade de cosseno entre o modelo do servidor e o modelo do cliente {client_id}: {cosine_similarity}")



'''
#______________________________________________________
model1 = LSTMModel(input_size=78, hidden_size=16, num_layers=5, output_size=2)
model2 = LSTMModel(input_size=78, hidden_size=16, num_layers=5, output_size=2)


model1.load_state_dict(torch.load("../results/cic-unb-models/dataset_1.pth"))
model2.load_state_dict(torch.load("../results/cic-unb-models/dataset_4.pth"))




# Comparar os pesos e bias de cada camada
for name1, param1 in model1.named_parameters():
    for name2, param2 in model2.named_parameters():
        if name1 == name2: # Se os nomes das camadas forem iguais
            print(param1)
            if torch.equal(param1, param2): # Se os parâmetros forem iguais
                print(f"A camada {name1} é igual nos dois modelos")
            else: # Se os parâmetros forem diferentes
                print(f"A camada {name1} é diferente nos dois modelos")
'''