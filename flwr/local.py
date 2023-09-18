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
import optuna
from optuna.trial import TrialState
import os
import torch.optim as optim

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
        # Duas camadas de RNN
        self.rnn1 = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.rnn2 = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True, dropout=0.2)

        # Camada de saída totalmente conectada
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Propagação através da primeira camada RNN
        out1, _ = self.rnn1(x)

        # Propagação através da segunda camada RNN
        out2, _ = self.rnn2(out1)
        out = self.fc(out2)
        return out






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


    num_layers = trial.suggest_int("num_layers", 5, 100, 100)  # Number of neurons of FC1 layer
    hidden_size = trial.suggest_int("hidden_size", 32, 64, 128)     # Dropout for convolutional layer 2


    # Generate the model
    model = LSTMModel(input_size=49, hidden_size=hidden_size, num_layers=num_layers, output_size=32).to(DEVICE)


    # Generate the optimizers
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])  # Optimizers
    lr = trial.suggest_float("lr", 0.0001, 0.1, log=True) # Learning rates
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Training of the model
    for epoch in range(n_epochs):
        train(model, optimizer)  # Train the model
        accuracy = test(model)   # Evaluate the model

        # For pruning (stops trial early if not promising)
        trial.report(accuracy, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

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


def train(net, optimizer):
    """Train the model on the training set."""

    losses = []  # Lista para armazenar os valores de perda
    accuracies = []  # Lista para armazenar os valores de acurácia
    criterion = torch.nn.CrossEntropyLoss()
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




def test(net):
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
    X_train, X_test, y_train, y_test = load_dataset(args.dataset_id)
    train_loader, val_loader = create_loaders(X_train, X_test, y_train, y_test)
    return train_loader, val_loader


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
#net = LSTMModel(input_size=49, hidden_size=128, num_layers=100, output_size=2).to(DEVICE)
#trainloader, testloader = load_data()
#train(net, trainloader, 200)
#test(net, testloader)



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
    df.to_csv('./results/'+str(args.dataset_id)+'_optuna_results.csv', index=False)  # Save to csv file

    # Display results in a dataframe
    print("\nOverall Results (ordered by accuracy):\n {}".format(df))

    # Find the most important hyperparameters
    most_important_parameters = optuna.importance.get_param_importances(study, target=None)

    # Display the most important hyperparameters
    print('\nMost important hyperparameters:')
    for key, value in most_important_parameters.items():
        print('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))
