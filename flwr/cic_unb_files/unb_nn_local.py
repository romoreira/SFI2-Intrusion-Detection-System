import warnings
from collections import OrderedDict
import time
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
import optuna
from optuna.trial import TrialState
import os
import torch.optim as optim
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

def objective(trial):
    """Objective function to be optimized by Optuna.

    Hyperparameters chosen to be optimized: optimizer, learning rate,
    dropout values, number of convolutional layers, number of filters of
    convolutional layers, number of neurons of fully connected layers.

    Inputs:
        - trial (optuna.trial._trial.Trial): Optuna trial
    Returns:
        - accuracy(torch.Tensor): The test accuracy. Parameter to be maximized.
    """


    #num_layers = trial.suggest_int("num_layers", 5, 30, 100)  # Number of neurons of FC1 layer 5, 100, 100
    #hidden_size = trial.suggest_int("hidden_size", 32, 64, 128)     # Dropout for convolutional layer 2
    num_layers = 5
    hidden_size = 16

    # Generate the model input_size 49
    model = LSTMModel(input_size=78, hidden_size=hidden_size, num_layers=num_layers, output_size=2).to(DEVICE)


    # Generate the optimizers
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])  # Optimizers
    lr = trial.suggest_float("lr", 0.0001, 0.001, log=True) # Learning rates 0.0001, 0.1
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Training of the model
    for epoch in range(10):
        train(model, optimizer, trainloader)  # Train the model
        accuracy = test(model, testloader)   # Evaluate the model

        # For pruning (stops trial early if not promising)
        trial.report(accuracy, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


def remove_spaces(column_name):
    return column_name.strip()

def load_dataset():
    # Lista de caminhos para os conjuntos de dados
    dataset_paths = [
        "../../dataset/cic-unb-ids/Tuesday-WorkingHours.pcap_ISCX.csv",
        "../../dataset/cic-unb-ids/Wednesday-workingHours.pcap_ISCX.csv",
        "../../dataset/cic-unb-ids/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "../../dataset/cic-unb-ids/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "../../dataset/cic-unb-ids/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "../../dataset/cic-unb-ids/Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "../../dataset/cic-unb-ids/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    ]

    # Lista para armazenar os DataFrames
    dfs = []

    # Loop para ler e concatenar os DataFrames
    for path in dataset_paths:
        df = pd.read_csv(path)
        dfs.append(df)

    # Concatenar DataFrames
    df = pd.concat(dfs, ignore_index=True)


    df.columns = df.columns.map(remove_spaces)

    df['Label'] = df['Label'].apply(lambda x: 'MALIGNANT' if x != 'BENIGN' else x)
    print(df['Label'].value_counts())

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


def train(net, optimizer, trainloader):
    losses = []  # Lista para armazenar os valores de perda
    accuracies = []  # Lista para armazenar os valores de acurácia
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    end_time = 0
    start_time = time.time()
    for epoch in range(10):
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

    end_time = time.time()

    # Plotar o gráfico de Loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, int(len(losses) + 1)), losses, label='Loss', linewidth=2, color='orange')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.legend()
    plt.grid(False)  # Desativa as gridlines
    plt.savefig(str("../results/cic-unb/") + str("joint_datasets_TRAIN_LOSS.pdf"))

    # Plotar o gráfico de Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, int(len(accuracies) + 1)), accuracies, label='Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.legend()
    plt.grid(False)  # Desativa as gridlines
    plt.savefig(str("../results/cic-unb/joint_datasets__TRAIN_ACC.pdf"))

    with open("../results/cic-unb/logs/trainig_time_joint_datasets.txt", 'a') as f:
    # Agora, quando você usa a função print com o argumento file, a saída será escrita no arquivo
        print(f"Total Training Time: {end_time - start_time} seconds - dataset_id: " + str(args.dataset_id), file=f)


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
    return accuracy


def load_data():
    """Load Custom Dataset."""
    X_train, X_test, y_train, y_test = load_dataset()
    train_loader, val_loader = create_loaders(X_train, X_test, y_train, y_test)
    return train_loader, val_loader


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = LSTMModel(input_size=78, hidden_size=16, num_layers=5, output_size=2).to(DEVICE)
trainloader, testloader = load_data()

# Training of the model

if str(args.optim) == "Adam":
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
elif str(args.optim) == "SGD":
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
elif str(args.optim) == "RMSprop":
    optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, momentum=0.9)

train(net, optimizer, trainloader)  # Train the model
accuracy = test(net, testloader)  # Evaluate the model

#torch.save(net.state_dict(), "../results/cic-unb-models/local_training_dataset_"+str(args.dataset_id)+".pth")


'''
if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Optimization study for a PyTorch CNN with Optuna
    # -------------------------------------------------------------------------

    # Use cuda if available for faster computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Parameters ----------------------------------------------------------
    n_epochs = 50                         # Number of training epochs
    batch_size_train = 64                 # Batch size for training data
    batch_size_test = 1000                # Batch size for testing data
    number_of_trials = 100                # Number of Optuna trials
    limit_obs = True                      # Limit number of observations for faster computation

    # *** Note: For more accurate results, do not limit the observations.
    #           If not limited, however, it might take a very long time to run.
    #           Another option is to limit the number of epochs. ***

    if limit_obs:  # Limit number of observations
        number_of_train_examples = 500 * batch_size_train  # Max train observations
        number_of_test_examples = 5 * batch_size_test      # Max test observations
    else:
        number_of_train_examples = 60000                   # Max train observations
        number_of_test_examples = 10000                    # Max test observations
    # -------------------------------------------------------------------------

    # Make runs repeatable
    random_seed = 1
    torch.backends.cudnn.enabled = False  # Disable cuDNN use of nondeterministic algorithms
    torch.manual_seed(random_seed)

    trainloader, testloader = load_data()

    # Create an Optuna study to maximize test accuracy
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=number_of_trials)

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    # Find number of pruned and completed trials
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Display the study statistics
    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial = study.best_trial
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save results to csv file
    df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'duration'], axis=1)  # Exclude columns
    df = df.loc[df['state'] == 'COMPLETE']        # Keep only results that did not prune
    df = df.drop('state', axis=1)                 # Exclude state column
    df = df.sort_values('value')                  # Sort based on accuracy
    df.to_csv('../results/cic-unb/optimization/'+str(args.dataset_id)+'_optuna_results.csv', index=False)  # Save to csv file

    # Add best trial value and params to the dataframe
    df['best_trial_value'] = trial.value
    for key, value in trial.params.items():
        df['best_trial_param_{}'.format(key)] = value

    df.to_csv('../results/cic-unb/optimization/' + str(args.dataset_id) + '_optuna_results.csv', mode='a', header=False, index=False)  # Save to csv file

    # Display results in a dataframe
    print("\nOverall Results (ordered by accuracy):\n {}".format(df))

    # Find the most important hyperparameters
    most_important_parameters = optuna.importance.get_param_importances(study, target=None)

    # Display the most important hyperparameters
    print('\nMost important hyperparameters:')
    for key, value in most_important_parameters.items():
        print('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))
'''