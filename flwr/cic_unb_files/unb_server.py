from typing import List, Tuple
import time
import flwr as fl
from flwr.common import Metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from typing import Optional
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Distbelief training example')
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=str, default='3002')
parser.add_argument('--world_size', type=int)
parser.add_argument('--rank', type=int)
parser.add_argument('--dataset', type=str, help='Nome do diretório do dataset')
parser.add_argument("--epochs", type=int)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--dataset_id", type=int, help='ID do DataSet')
parser.add_argument("--batch_size", type=int, default=32, help='Batch Size do Dataset')
args = parser.parse_args()


'''
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x
'''

#CNN de Teste para o problema binário
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.dropout(x)  # Aplicar o dropout após a ativação
        x = self.output_layer(x)
        # Aplicar a função Softmax na camada de saída
        x = nn.functional.softmax(x, dim=1)
        return x


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_sample = self.X[idx]
        y_sample = self.y[idx]
        return x_sample, y_sample

def column_remover(dataframe):
    colunas_para_remover = ['sAddress', 'rAddress', 'sMACs', 'rMACs', 'sIPs', 'rIPs', 'protocol', 'startDate',
                            'endDate', 'start', 'end']


    #Remover a rotulagem: Injection Timing
    colunas_para_remover.append("IT_B_Label")
    colunas_para_remover.append("IT_M_Label")
    colunas_para_remover.append("NST_M_Label")

    # Verifique se as colunas a serem removidas existem no DataFrame
    colunas_existentes = dataframe.columns.tolist()
    colunas_para_remover = [coluna for coluna in colunas_para_remover if coluna in colunas_existentes]

    dataframe = dataframe.drop(columns=colunas_para_remover)
    #print("TYPE: "+str(type(dataframe)))
    #print("Coluns: "+str(dataframe.columns))
    #print(dataframe.dtypes)
    #exit()
    # Remove as colunas especificadas

    #Preencher os valores do DF quando houver algumas lacunas (em branco)
    dataframe.fillna(0, inplace=True)
    #print(dataframe.head)
    #print("Quantidade de Colunas Total: "+str(len(dataframe.columns)))
    #print("Coluna NST_B_Label: "+str(dataframe['NST_M_Label']))
    print("Dataset cleaned!")
    return dataframe

def remove_spaces(column_name):
    return column_name.strip()

def create_federated_testloader(dataset_id):
    if dataset_id == 1:
        # Caminho para o diretório do conjunto de dados
        data_dir = "../../dataset/cic-unb-ids/Tuesday-WorkingHours.pcap_ISCX.csv"
    elif dataset_id == 2:
        # Caminho para o diretório do conjunto de dados
        data_dir = "../../dataset/cic-unb-ids/Wednesday-workingHours.pcap_ISCX.csv"
    elif dataset_id == 3:
        # Caminho para o diretório do conjunto de dados
        data_dir = "../../dataset/cic-unb-ids/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
    elif dataset_id == 4:
        # Caminho para o diretório do conjunto de dados
        data_dir = "../../dataset/cic-unb-ids/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    elif dataset_id == 5:
        # Caminho para o diretório do conjunto de dados
        data_dir = "../../dataset/cic-unb-ids/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
    elif dataset_id == 6:
        # Caminho para o diretório do conjunto de dados
        data_dir = "../../dataset/cic-unb-ids/Friday-WorkingHours-Morning.pcap_ISCX.csv"
    elif dataset_id == 7:
        # Caminho para o diretório do conjunto de dados
        data_dir = "../../dataset/cic-unb-ids/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

    df = pd.read_csv(data_dir)
    # df = column_remover(df)
    df.columns = df.columns.map(remove_spaces)

    df['Label'] = df['Label'].apply(lambda x: 'MALIGNANT' if x != 'BENIGN' else x)

    # Separar as colunas de recursos (features) e rótulos (labels)
    X = df.drop('Label', axis=1)  # Substitua 'label' pelo nome da coluna de rótulos
    y = df['Label']

    # Dividir os dados em conjuntos de treinamento e teste
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Crie um objeto LabelEncoder
    label_encoder = LabelEncoder()

    # Ajuste o LabelEncoder aos seus dados de classe (y_train)
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.fit_transform(y_test)

    # Crie um tensor PyTorch a partir dos valores codificados
    y_train = torch.tensor(y_train_encoded, dtype=torch.int64)
    y_test = torch.tensor(y_test_encoded, dtype=torch.int64)

    # Padronizar os recursos (opcional, mas geralmente recomendado)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Converter os dados para tensores PyTorch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    # y_train = torch.tensor(y_train.values, dtype=torch.int64)
    # y_test = torch.tensor(y_test.values, dtype=torch.int64)

    test_dataset = CustomDataset(X_test, y_test)

    # Crie os data loaders
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return '', test_loader

def create_loaders(X_train, X_test, y_train, y_test):
    #print("\n")
    #print("Shape X_train: "+str(X_train.shape))
    #print("Shape X_test: " + str(X_test.shape))
    #print("Shape y_train: " + str(y_train.shape))
    #print("Shape Y_test: " + str(y_test.shape))
    #exit()
    print("TYPE: "+str(type(X_test)))
    print("TYPE: " + str(type(y_test)))
    # Crie conjuntos de dados para treinamento e teste
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    # Crie os data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader

def load_dataset(dataset_id):


    if dataset_id == 1:
        # Caminho para o diretório do conjunto de dados
        data_dir = "../dataset/extended/output_bottom.csv"
    elif dataset_id == 2:
        # Caminho para o diretório do conjunto de dados
        data_dir = "../dataset/extended/output_left.csv"
    elif dataset_id == 3:
        # Caminho para o diretório do conjunto de dados
        data_dir = "../dataset/extended/output_right.csv"

    df = pd.read_csv(data_dir)
    df = column_remover(df)

    # Separar as colunas de recursos (features) e rótulos (labels)
    X = df.drop('NST_B_Label', axis=1)  # Substitua 'label' pelo nome da coluna de rótulos
    y = df['NST_B_Label']

    # Dividir os dados em conjuntos de treinamento e teste
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Padronizar os recursos (opcional, mas geralmente recomendado)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Converter os dados para tensores PyTorch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.int64)
    y_test = torch.tensor(y_test.values, dtype=torch.int64)
    return X_train, X_test, y_train, y_test

def load_data():
    """Load Custom Dataset."""
    X_train, X_test, y_train, y_test = load_dataset(args.dataset_id)
    train_loader, val_loader = create_loaders(X_train, X_test, y_train, y_test)
    return train_loader, val_loader

def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for X, y in tqdm(testloader):
            outputs = net(X.to(DEVICE))
            y = y.to(DEVICE)
            loss += criterion(outputs, y).item()
            correct += (torch.max(outputs.data, 1)[1] == y).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)




# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    print("Metrics: "+str(metrics))
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    print("Acuracies list: "+str(len(accuracies)))
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    print(f"\n### Server-side evaluation accuracy: "+str(float(sum(accuracies) / sum(examples))))
    return {"accuracy": sum(accuracies) / sum(examples)}

_, testloader = create_federated_testloader(1)
net = LSTMModel(input_size=78, hidden_size=16, num_layers=5, output_size=2).to(DEVICE)

# The `evaluate` function will be by Flower called after every round
def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    net = LSTMModel(input_size=78, hidden_size=16, num_layers=5, output_size=2).to(DEVICE)
    acc = []

    _, testloader = create_federated_testloader(1)
    loss, accuracy = test(net, testloader)
    set_parameters(net, parameters)
    torch.save(net.state_dict(), "../results/cic-unb-models/server_model_aggregated.pth")

    datasets = [1, 2, 3, 4, 5, 6, 7]
    for i in datasets:
        _, testloader = create_federated_testloader(i)
        loss, accuracy = test(net, testloader)
        accuracy_percent = accuracy * 100  # Multiplica a precisão por 100 para obter o valor percentual
        acc.append(accuracy_percent)
        #print(f"\n### Loss {loss} and accuracy {accuracy_percent:.2f}% using DatasetID: {i} ###\n")
    print(f"\n## Final Server-Side Acc: "+str((sum(acc) / len(acc))))
    return loss, {"accuracy": (sum(acc)/len(acc))}

def default_evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    net = LSTMModel(input_size=78, hidden_size=16, num_layers=5, output_size=2).to(DEVICE)
    _, valloader = create_federated_testloader(1)
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, valloader)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}


def simple_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Calculate the average accuracy of each client
    accuracies = [m["accuracy"] for _, m in metrics]
    average_accuracy = sum(accuracies) / len(accuracies)

    # Return the average accuracy as the evaluation result
    return {"accuracy": average_accuracy}

# Define strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.3,
    fraction_evaluate=0.3,
    min_fit_clients=7,
    min_evaluate_clients=7,
    min_available_clients=7,
    #evaluate_fn=evaluate,
    evaluate_fn=default_evaluate,
    #evaluate_fn=simple_average,
    #evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
)



# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=2),
    strategy=strategy
)