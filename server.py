import os
import sys
import argparse
from torch import nn
from fedlab.core.network import DistNetwork
from fedlab.contrib.algorithm.basic_server import AsyncServerHandler
from fedlab.core.server.manager import AsynchronousServerManager
from fedlab.models import MLP
import torch
from torchvision import datasets, models, transforms
import torchvision
import torchvision.transforms as transforms
import random
import torch
from torchvision import datasets, models, transforms
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import pandas as pd

#CNN de Teste para o problema binário
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        #self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        #x = self.sigmoid(x)
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

def create_loaders(X_train, X_test, y_train, y_test ):
    # Crie conjuntos de dados para treinamento e teste
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    # Defina tamanhos de lote (batch sizes)


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
    data_dir = ''
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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


def evaluate_model(val_loader, model):
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
    print(f'Acurácia no conjunto de validação: {100 * accuracy:.2f}%')

    # Agora, você pode imprimir os rótulos verdadeiros e previstos
    #print("Rótulos Verdadeiros:", real_labels)
    #print("Rótulos Previstos:", predicted_labels)

def select_model(name):
    # Basic CNN
    if name == 'Basic':
        # Set hyperparameters
        input_size = 49  # Number of features in your dataset
        hidden_size = 64  # Number of neurons in the hidden layer
        output_size = 2  # 1 for binary classification

        # Initialize the model
        model = SimpleNN(input_size, hidden_size, output_size)
        return model

    # ShuffleNet
    if name == 'ShuffleNet':
        model_ft = models.shufflenet_v2_x1_0(pretrained=True)
        # Congele todos os parâmetros do modelo (ou deixe-os descongelados, dependendo de sua escolha)
        for param in model_ft.parameters():
            param.requires_grad = True  # Defina como False para congelar
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        input_size = 224
        return model_ft

    elif name == 'AlexNet':
        ###AlexNet###
        feature_extract = True
        model_ft = models.alexnet(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, 4)
        input_size = 224
        return model_ft

    elif name == 'Resnet18':
        ###Resnet18###
        feature_extract = True
        model_ft = models.resnet18(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 4)
        input_size = 224

    elif name == 'SqueezeNet':
        ###SqueezeNet###
        model_ft = models.squeezenet1_0(pretrained=True)
        set_parameter_requires_grad(model_ft, True)
        model_ft.classifier[1] = nn.Conv2d(512, 4, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = 4
        input_size = 224
        return model_ft

    elif name == 'VGG11_b':
        ###VGG11_b###
        use_pretrained = True
        feature_extract = True
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, 4)
        input_size = 224
        return model_ft

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')

    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=str, default='3002')
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--dataset', type=str, help='Nome do diretório do dataset')
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--dataset_id", type=int, help='ID do DataSet')
    parser.add_argument("--batch_size", type=int, default=32, help='Batch Size do Dataset')
    args = parser.parse_args()

    if torch.cuda.is_available():
        args.cuda = True
    else:
        args.cuda = False


    model = select_model('Basic')


    handler = AsyncServerHandler(model, global_round=5)
    handler.setup_optim(0.5)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0)
    Manager = AsynchronousServerManager(handler=handler, network=network)

    Manager.run()

    X_train, X_test, y_train, y_test = load_dataset(args.dataset_id)
    train_loader, val_loader = create_loaders(X_train, X_test, y_train, y_test)
    evaluate_model(val_loader, model)