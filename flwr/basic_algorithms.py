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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    X = X.values
    y = y.values

    # Dividir os dados em conjuntos de treinamento e teste
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return X_train, X_test, y_train, y_test

def save_accuracy_to_txt(acc_list, algorithm_name, dataset):
    """
    Salva os valores da lista de acurácias em um arquivo de texto.

    Args:
        acc_list (list): Lista de acurácias a serem salvas.
        algorithm_name (str): Nome do algoritmo associado às acurácias.
        output_file (str): Nome do arquivo de saída (com extensão .txt).

    Returns:
        None
    """
    try:
        with open("./results/basic_algoritms/ds_"+str(dataset)+"_"+str(algorithm_name), 'w') as file:
            file.write(f"Acurácias para o algoritmo {algorithm_name}:\n")
            for i, acc in enumerate(acc_list):
                file.write(f"Fold {i+1}: {acc:.2f}\n")
        print(f"Acurácias salvas em '{algorithm_name}' com sucesso.")
    except Exception as e:
        print(f"Erro ao salvar as acurácias: {str(e)}")

def svm(X_train, X_test, y_train, y_test):
    # 1. Inicialize o classificador SVM
    svm_classifier = SVC()  # Você pode ajustar os hiperparâmetros aqui, se necessário

    # 2. Treine o modelo SVM com os dados de treinamento
    svm_classifier.fit(X_train, y_train)

    # 3. Faça previsões com o modelo treinado
    y_pred = svm_classifier.predict(X_test)

    # 4. Avalie o desempenho do modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia do modelo: {accuracy:.2f}')

    # 5. Você também pode gerar um relatório de classificação
    report = classification_report(y_test, y_pred)
    print('Relatório de Classificação:')
    print(report)

    return accuracy * 100

def knn(X_train, X_test,y_train, y_test):

    # 4. Inicialize o classificador KNN (defina o valor de 'n_neighbors' de acordo com suas necessidades)
    knn = KNeighborsClassifier(n_neighbors=5)  # Você pode ajustar o valor de 'n_neighbors'

    # 5. Treine o modelo KNN com os dados de treinamento
    knn.fit(X_train, y_train)

    # 6. Faça previsões com o modelo treinado
    y_pred = knn.predict(X_test)

    # 7. Avalie o desempenho do modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia do modelo: {accuracy:.2f}')

    # Você também pode gerar um relatório de classificação
    report = classification_report(y_test, y_pred)
    print('Relatório de Classificação:')
    print(report)
    return accuracy * 100

def random_forest(X_train, X_test, y_train, y_test):
    # 1. Inicialize o classificador Random Forest
    rf = RandomForestClassifier()  # Você pode ajustar os hiperparâmetros aqui, se necessário

    # 2. Treine o modelo Random Forest com os dados de treinamento
    rf.fit(X_train, y_train)

    # 3. Faça previsões com o modelo treinado
    y_pred = rf.predict(X_test)

    # 4. Avalie o desempenho do modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia do modelo: {accuracy:.2f}')

    # 5. Você também pode gerar um relatório de classificação
    report = classification_report(y_test, y_pred)
    print('Relatório de Classificação:')
    print(report)

    return accuracy * 100

def decision_tree(X_train, X_test, y_train, y_test):
    # 1. Inicialize o classificador da Árvore de Decisão
    dt = DecisionTreeClassifier()  # Você pode ajustar os hiperparâmetros aqui, se necessário

    # 2. Treine o modelo de Árvore de Decisão com os dados de treinamento
    dt.fit(X_train, y_train)

    # 3. Faça previsões com o modelo treinado
    y_pred = dt.predict(X_test)

    # 4. Avalie o desempenho do modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia do modelo: {accuracy:.2f}')

    # 5. Você também pode gerar um relatório de classificação
    report = classification_report(y_test, y_pred)
    print('Relatório de Classificação:')
    print(report)

    return accuracy * 100

X_train, X_test, y_train, y_test = load_dataset(1)
datasets = [1, 2, 3]

for dataset in range(len(datasets)):
    #KNN
    acc_list = []
    for i in range(10):
        acc_list.append(knn(X_train, X_test, y_train, y_test))
    save_accuracy_to_txt(acc_list, "KNN", dataset=dataset+1)

    #Decision Tree
    acc_list = []
    for i in range(10):
        acc_list.append(decision_tree(X_train, X_test, y_train, y_test))
    save_accuracy_to_txt(acc_list, "DecisionTree", dataset=dataset+1)

    # RF
    acc_list = []
    for i in range(10):
        acc_list.append(random_forest(X_train, X_test, y_train, y_test))
    save_accuracy_to_txt(acc_list, "Random Forest", dataset=dataset + 1)

    # SVM
    acc_list = []
    for i in range(10):
        acc_list.append(svm(X_train, X_test, y_train, y_test))
    save_accuracy_to_txt(acc_list, "SVM", dataset=dataset + 1)