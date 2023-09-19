from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

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

def create_loaders(X_train, X_test, y_train, y_test):
    #print("\n")
    #print("Shape X_train: "+str(X_train.shape))
    #print("Shape X_test: " + str(X_test.shape))
    #print("Shape y_train: " + str(y_train.shape))
    #print("Shape Y_test: " + str(y_test.shape))
    #exit()



    # Crie conjuntos de dados para treinamento e teste
    #train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    # Crie os data loaders
    #train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return '', test_loader

def load_dataset():

    dataset_id = ''
    data_dir = ''
    size = 0
    for i in range(3):
        dataset_id = int(i) + 1
        print("Dataset ID: "+str(i))
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
        resultado = pd.concat([X_test, y_test], axis=1)
        resultado.to_csv('../dataset/extended_fed_tester/test_'+str(dataset_id)+'.csv', index=False)

        size += len(resultado)
    print("Tamanho Médio:> "+str(int(size/3)))

def build_testset():
    import pandas as pd
    import random

    # Carregue os três CSVs em DataFrames separados
    df1 = pd.read_csv('../dataset/extended_fed_tester/test_1.csv')
    df2 = pd.read_csv('../dataset/extended_fed_tester/test_2.csv')
    df3 = pd.read_csv('../dataset/extended_fed_tester/test_3.csv')

    df1_embaralhado = df1.sample(frac=1).reset_index(drop=True)
    df2_embaralhado = df2.sample(frac=1).reset_index(drop=True)
    df3_embaralhado = df3.sample(frac=1).reset_index(drop=True)

    print("Tamanho DS1: "+str(len(df1_embaralhado)))
    print("Tamanho DS2: " + str(len(df2_embaralhado)))
    print("Tamanho DS3: " + str(len(df3_embaralhado)))

    # Determine o tamanho desejado do novo DataFrame
    tamanho_desejado = 2291

    # Inicialize o novo DataFrame
    novo_df = pd.DataFrame(columns=df1.columns)

    # Crie uma lista com os DataFrames originais
    dataframes = [df1, df2, df3]

    # Itere até que o novo DataFrame alcance o tamanho desejado
    while len(novo_df) < tamanho_desejado:
        # Escolha aleatoriamente um dos DataFrames
        dataframe_escolhido = random.choice(dataframes)

        # Escolha aleatoriamente uma linha desse DataFrame
        linha_escolhida = dataframe_escolhido.sample(n=1)

        # Adicione a linha ao novo DataFrame
        novo_df = pd.concat([novo_df, linha_escolhida], ignore_index=True)
    print("Dataframe_FINAL: "+str(len(novo_df)))
    novo_df.to_csv('../dataset/extended_fed_tester/dataframe_final.csv', index=False)


def create_federated_testloader():
    # Leia o arquivo CSV em um DataFrame
    df = pd.read_csv('../dataset/extended_fed_tester/dataframe_final.csv')

    # Crie o primeiro DataFrame com as primeiras 49 colunas
    X_test = df.iloc[:, :49]

    # Crie o segundo DataFrame com a última coluna
    y_test = df.iloc[:, -1]

    X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.int64)


    test_dataset = CustomDataset(X_test, y_test)

    # Crie os data loaders
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return test_loader

load_dataset()
build_testset()
create_federated_testloader()
