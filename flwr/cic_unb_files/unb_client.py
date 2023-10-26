import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.dropout(x)  # Aplicar o dropout após a ativação
        x = self.output_layer(x)
        # Aplicar a função Softmax na camada de saída
        x = nn.functional.softmax(x, dim=1)
        return x

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
parser.add_argument("--optim", type=str, help='Optimizer to choose: Adam or SGD')
args = parser.parse_args()

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

def load_dataset(dataset_id):


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
        #data_dir = "../../dataset/cic-unb-ids/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
        data_dir = ""
    elif dataset_id == 5:
        # Caminho para o diretório do conjunto de dados
        data_dir = "../../dataset/cic-unb-ids/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
    elif dataset_id == 6:
        # Caminho para o diretório do conjunto de dados
        #data_dir = "../../dataset/cic-unb-ids/Friday-WorkingHours-Morning.pcap_ISCX.csv"
        data_dir = ""
    elif dataset_id == 7:
        # Caminho para o diretório do conjunto de dados
        #data_dir = "../../dataset/cic-unb-ids/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
        data_dir = ""


    df = pd.read_csv(data_dir)
    #df = column_remover(df)
    df.columns = df.columns.map(remove_spaces)

    df['Label'] = df['Label'].apply(lambda x: 'MALIGNANT' if x != 'BENIGN' else x)

    # Separar as colunas de recursos (features) e rótulos (labels)
    X = df.drop('Label', axis=1)  # Substitua 'label' pelo nome da coluna de rótulos
    y = df['Label']

    # Dividir os dados em conjuntos de treinamento e teste
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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
    #y_train = torch.tensor(y_train.values, dtype=torch.int64)
    #y_test = torch.tensor(y_test.values, dtype=torch.int64)

    return X_train, X_test, y_train, y_test



def create_loaders(X_train, X_test, y_train, y_test):
    #print("\n")
    #print("Shape X_train: "+str(X_train.shape))
    #print("Shape X_test: " + str(X_test.shape))
    #print("Shape y_train: " + str(y_train.shape))
    #print("Shape Y_test: " + str(y_test.shape))
    #exit()

    # Crie conjuntos de dados para treinamento e teste
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    # Crie os data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    print("Starting Client training for " + str(epochs) + " epochs")
    criterion = torch.nn.CrossEntropyLoss()


    if str(args.optim) == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    elif str(args.optim) == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    elif str(args.optim) == "RMSprop":
        optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, momentum=0.9)

    losses = []  # Lista para armazenar os valores de perda
    accuracies = []  # Lista para armazenar os valores de acurácia
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for X, y in tqdm(trainloader):
            optimizer.zero_grad()
            loss = criterion(net(X.to(DEVICE)), y.to(DEVICE))
            loss.backward()
            optimizer.step()
            epoch_loss += loss

            # Atualize o número de previsões corretas e o total
            _, predicted = torch.max(net(X.to(DEVICE)).data, 1)
            total += y.size(0)
            correct += (predicted == y.to(DEVICE)).sum().item()

        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total  # Calcule a acurácia aqui, dentro do loop externo
        print(f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy: {round(float(epoch_acc) * 100, 2)}%")

        losses.append(epoch_loss.item())  # Adicione o valor de perda à lista
        accuracies.append(epoch_acc)  # Adicione o valor de acurácia à lista

    # Plotar o gráfico de Loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, int(len(losses) + 1)), losses, label='Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(False)  # Desativa as gridlines
    plt.savefig(str("../results/cic-unb/")+str("client_")+str(args.dataset_id)+"_TRAIN_LOSS.pdf")

    # Plotar o gráfico de Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, int(len(accuracies) + 1)), accuracies, label='Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(False)  # Desativa as gridlines
    plt.savefig(str("../results/cic-unb/")+str("client_")+str(args.dataset_id)+"_TRAIN_ACC.pdf")


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


def load_data():
    """Load Custom Dataset."""
    X_train, X_test, y_train, y_test = load_dataset(args.dataset_id)
    train_loader, val_loader = create_loaders(X_train, X_test, y_train, y_test)
    return train_loader, val_loader


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################


net = LSTMModel(input_size=78, hidden_size=16, num_layers=5, output_size=2).to(DEVICE)
trainloader, testloader = load_data()


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=args.epochs)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        print("Acurácia do Cliente: "+str(args.dataset_id)+str(" eh: ")+str(accuracy))
        return float(loss), len(testloader.dataset), {"accuracy": round(float(accuracy) * 100, 2)}



# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)
