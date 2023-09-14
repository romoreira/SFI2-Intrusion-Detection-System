import torchvision
import torchvision.transforms as transforms
import torch
import argparse
import sys
import os
from torch import nn
import pandas as pd
sys.path.append("../../")
from fedlab.core.client.manager import ActiveClientManager
from fedlab.core.network import DistNetwork
from fedlab.contrib.algorithm.basic_client import SGDClientTrainer
from fedlab.models import MLP
from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST
import torch
from torchvision import datasets, models, transforms
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.aggregator import Aggregators
from tqdm import trange
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

#CNN de Teste para o problema binário
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers=16, batch_first=True)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        #x = self.lstm(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
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

def evaluate_model(val_loader, model, client_id):
    total = 0
    correct = 0
    real_labels = []  # Lista para armazenar os rótulos verdadeiros
    predicted_labels = []  # Lista para armazenar os rótulos previstos

    with torch.no_grad():
        model.eval()

        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Supondo classificação binária
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Adicione os rótulos verdadeiros e previstos às listas
            real_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    accuracy = correct / total
    print(f'Client: '+str(client_id)+f' com Acurácia no conjunto de validação: {100 * accuracy:.2f}%')

    # Agora, você pode imprimir os rótulos verdadeiros e previstos
    #print("Rótulos Verdadeiros:", real_labels)
    #print("Rótulos Previstos:", predicted_labels)

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


    return dataframe

def load_dataset(dataset_id):

    if dataset_id == 1:
        # Caminho para o diretório do conjunto de dados
        data_dir = "./dataset/extended/output_bottom.csv"
    elif dataset_id == 2:
        # Caminho para o diretório do conjunto de dados
        data_dir = "./dataset/extended/output_left.csv"
    elif dataset_id == 3:
        # Caminho para o diretório do conjunto de dados
        data_dir = "./dataset/extended/output_right.csv"

    df = pd.read_csv(data_dir)
    df = column_remover(df)

    # Separar as colunas de recursos (features) e rótulos (labels)
    X = df.drop('NST_B_Label', axis=1)  # Substitua 'label' pelo nome da coluna de rótulos
    y = df['NST_B_Label']

    # Dividir os dados em conjuntos de treinamento e teste
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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


def select_model(name):
    # Basic CNN
    if name == 'Basic':
        # Set hyperparameters
        input_size = 49  # Number of features in your dataset
        hidden_size = 32  # Number of neurons in the hidden layer
        output_size = 2  # 1 for binary classification

        # Initialize the model
        model = SimpleNN(input_size, hidden_size, output_size)
        return model



class FedAvgClientTrainer(SGDClientTrainer):
    def __init__(self, model, criterion):
        super().__init__(model, criterion)
        self.time = 0
        #self.round = 0

    @property
    def uplink_package(self):
        return [self.model_parameters, self.round]


    def local_process(self, payload, id):
        model_parameters = payload[0]
        self.round = payload[0]
        #train_loader = self.dataset.get_dataloader(id, self.batch_size)
        train_loader = self.dataset
        self.train(model_parameters, train_loader)

    def train(self, model_parameters, train_loader):
        """Client trains its local model on local dataset.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
        """
        SerializationTool.deserialize_model(self._model, model_parameters)  # load parameters
        self._LOGGER.info("Local train procedure is running")

        # Initialize a list to store loss values per epoch
        epoch_loss_values = []

        for ep in range(self.epochs):
            self._model.train()
            self._LOGGER.info("Client: " + str(args.rank) + f" Epoch {ep + 1}/{self.epochs}")

            # Initialize loss for the epoch
            epoch_loss = 0.0

            # Create a progress bar for the inner loop (batches)
            batch_iterator = tqdm(train_loader, desc='Training', leave=False)

            for data, target in batch_iterator:
                if self.cuda:
                    data, target = data.cuda(self.device), target.cuda(self.device)

                outputs = self._model(data)
                loss = self.criterion(outputs, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Accumulate the loss for the epoch
                epoch_loss += loss.item()

                # Update the progress bar
                batch_iterator.set_postfix({'Loss': loss.item()})

            # Append the average loss for the epoch to the list
            epoch_loss /= len(train_loader)
            epoch_loss_values.append(epoch_loss)

        self._LOGGER.info("Local train procedure is finished")

        # Plot the loss curve by epoch with integer values on the x-axis
        plt.figure(figsize=(10, 5))
        x_ticks = np.arange(1, len(epoch_loss_values) + 1)  # Generate integer ticks
        plt.plot(x_ticks, epoch_loss_values, label='Training Loss', linewidth=2.0)
        plt.xlabel('Epochs', fontsize=13)
        plt.ylabel('Loss', fontsize=13)
        plt.title('Training Loss', fontsize=15)
        plt.legend()
        plt.xticks(x_ticks)
        # plt.grid(True)
        plt.savefig(str(args.rank) + "_training_loss.pdf")


parser = argparse.ArgumentParser(description='Distbelief training example')
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=str, default='3002')
parser.add_argument('--world_size', type=int)
parser.add_argument('--rank', type=int)
parser.add_argument('--dataset', type=str, help='Nome do diretório do dataset')
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--dataset_id", type=int, help='ID do DataSet')
parser.add_argument("--batch_size", type=int, default=32, help='Batch Size do Dataset')
args = parser.parse_args()

if torch.cuda.is_available():
    args.cuda = True
else:
    args.cuda = False


model = select_model('Basic')


X_train, X_test, y_train, y_test = load_dataset(args.dataset_id)
train_loader, val_loader = create_loaders(X_train, X_test, y_train, y_test)

print("Client: "+str(args.rank)+" training with dataset: "+str(args.dataset_id))

criterion = nn.BCELoss()
trainer = FedAvgClientTrainer(model, criterion)


dataset = train_loader
print(dataset)
trainer.setup_dataset(dataset)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

network = DistNetwork(address=(args.ip, args.port),
                      world_size=args.world_size,
                      rank=args.rank)

Manager = ActiveClientManager(trainer=trainer, network=network)
Manager.run()
evaluate_model(val_loader, model, int(args.rank))