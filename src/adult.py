# Bibliotecas necessárias --------------------------------------------------------------------------------------------------------------------- #

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

# Estilizações -------------------------------------------------------------------------------------------------------------------------------- #

class Cores:
    RESET = "\033[0m"
    ROSA = "\033[38;2;255;105;180m"
    VERDE = "\033[38;2;144;238;144m"
    AZUL = "\033[38;2;135;206;250m"
    AMARELO = "\033[38;2;255;223;100m"
    ROXO = "\033[38;2;218;112;214m"
    PESSEGO = "\033[38;2;255;160;122m"

def print_quadro(titulo, cor=Cores.ROXO):
    largura = len(titulo) + 4
    print(cor + "+" + "-" * (largura - 2) + "+")
    print("| " + titulo + " |")
    print("+" + "-" * (largura - 2) + "+" + Cores.RESET)

# Manuseio dos dados --------------------------------------------------------------------------------------------------------------------------- #

def tratar_outliers(dados, coluna):
    q1 = dados[coluna].quantile(0.25)
    q3 = dados[coluna].quantile(0.75)
    iqr = q3 - q1
    inferior = q1 - 1.5 * iqr
    superior = q3 + 1.5 * iqr

    dados[coluna] = np.where(dados[coluna] < inferior, inferior, dados[coluna])
    dados[coluna] = np.where(dados[coluna] > superior, superior, dados[coluna])

# Lendo a base de dados

dados_csv = pd.read_csv('src/adult.csv')

dados_csv.replace('?', np.nan, inplace=True)
dados_csv.dropna(inplace=True)

num_cols = ['age','fnlwgt','educational-num','capital-gain','capital-loss','hours-per-week']
for coluna in num_cols:
    tratar_outliers(dados_csv, coluna)

# ------------------------------------------------------------------------------------------------------------------------------ #
# Funções para cada opção do menu

def painel_estatistica():
    print_quadro("Informações relativas à base de dados")
    print(dados_csv.head().to_string(index=False))

    valores_faltantes = dados_csv.isnull().sum().reset_index()
    valores_faltantes.columns = ['Coluna', 'Valores Faltantes']

    print("\n")
    print_quadro("Valores faltantes na base de dados:")
    header = f"{Cores.AMARELO}{valores_faltantes.columns[0]:<30}{valores_faltantes.columns[1]:>15}{Cores.RESET}"
    print(header)
    for index, row in valores_faltantes.iterrows():
        print(f"{row['Coluna']:<30}{row['Valores Faltantes']:>10}")

    print("\n")
    print_quadro("Panorama estatístico geral da base de dados:")
    print(dados_csv.describe())

    print("\n")
    num_cols = dados_csv.select_dtypes(include=[np.number]).columns
    estatisticas = pd.DataFrame({
        'Média': dados_csv[num_cols].mean(),
        'Mediana': dados_csv[num_cols].median(),
        'Variância': dados_csv[num_cols].var(),
        'Desvio Padrão': dados_csv[num_cols].std()
    })
    print_quadro("Resumo estatístico da base de dados:")
    print(f"{Cores.AZUL}{'Atributo':<20}{'Média':>15}{'Mediana':>15}{'Variância':>15}{'Desvio Padrão':>20}{Cores.RESET}")
    for index, row in estatisticas.iterrows():
        print(f"{index:<20}{row['Média']:>15.2f}{row['Mediana']:>15.2f}{row['Variância']:>15.2f}{row['Desvio Padrão']:>20.2f}")

def graficos_histograma():
    plt.figure(figsize=(8,6))
    sns.histplot(dados_csv['age'], bins=20, kde=True)
    plt.title("Histograma da idade")
    plt.xlabel("Idade")
    plt.ylabel("Frequência")
    plt.show()

    plt.figure(figsize=(8,6))
    sns.histplot(dados_csv['hours-per-week'], bins=20, kde=True, color='purple')
    plt.title("Distribuição de Horas Trabalhadas por Semana")
    plt.xlabel("Horas por Semana")
    plt.ylabel("Número de Pessoas")
    plt.show()

    plt.figure(figsize=(8,6))
    sns.histplot(dados_csv['educational-num'], bins=15, kde=True, color='orange')
    plt.title("Distribuição da Escolaridade (Número de Anos)")
    plt.xlabel("Número de Anos de Escolaridade")
    plt.ylabel("Número de Pessoas")
    plt.show()

def graficos_barras_agrupadas():
    palette_azul = ["#AEC6CF", "#779ECB"]
    plt.figure(figsize=(8,6))
    sns.countplot(data=dados_csv, x='gender', hue='income', palette=palette_azul)
    plt.title("Contagem de pessoas por sexo e renda")
    plt.show()

    plt.figure(figsize=(10,6))
    sns.countplot(data=dados_csv, x='race', hue='income', palette=palette_azul)
    plt.title("Renda por Raça")
    plt.xlabel("Raça")
    plt.ylabel("Número de Pessoas")
    plt.show()

def graficos_boxplot():
    plt.figure(figsize=(8,6))
    sns.boxplot(x='income', y='hours-per-week', data=dados_csv, showfliers=False)
    plt.title("Boxplot de horas por semana por faixa de renda")
    plt.show()

def grafico_impacto_variaveis():
    x = dados_csv[['age', 'educational-num', 'hours-per-week', 'occupation']].copy()
    y = dados_csv['income'].copy()

    le_income = LabelEncoder()
    y = le_income.fit_transform(y)

    le_occupation = LabelEncoder()
    x['occupation'] = le_occupation.fit_transform(x['occupation'])

    x = x.astype(float)
    y = y.astype(float)
    x = sm.add_constant(x)

    modelo = sm.Logit(y, x).fit()
    print("\n")
    print_quadro("Sumário")
    print(modelo.summary())

    labels = ['Idade', 'Escolaridade', 'Horas por semana', 'Ocupação']
    coeficientes = modelo.params[1:]

    plt.figure(figsize=(8,5))
    bars = plt.bar(labels, coeficientes, color=['#87CEFA', '#FFD700', '#FF69B4', '#DA70D6'])
    plt.title("Impacto das Variáveis na Renda (Coeficientes Logit)")
    plt.xlabel("Variáveis")
    plt.ylabel("Coeficiente")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.show()

# ------------------------------------------------------------------------------------------------------------------------------ #
# Menu principal

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def menu():
    while True:
        print_quadro("MENU PRINCIPAL", Cores.AZUL)
        print("1. Visualizar painel estatística")
        print("2. Plotar gráficos de histograma")
        print("3. Plotar gráficos de barras agrupadas")
        print("4. Plotar boxplots")
        print("5. Gráfico de barra para verificar o impactos de variavéis")
        print("0. Sair")

        opcao = input("Escolha uma opção: ")

        if opcao == "1":
            print("\n")
            painel_estatistica()
        elif opcao == "2":
            print(f"\n{Cores.ROSA}\tPlotando gráficos de histograma...{Cores.RESET}")
            graficos_histograma()
            print(f"\n{Cores.VERDE}\tGráfico(s) plotado(s) com sucesso!{Cores.RESET}")
        elif opcao == "3":
            print(f"\n{Cores.ROSA}\tPlotando gráficos de barras agrupadas...{Cores.RESET}")
            graficos_barras_agrupadas()
            print(f"\n{Cores.VERDE}\tGráfico(s) plotado(s) com sucesso!{Cores.RESET}")
        elif opcao == "4":
            print(f"\n{Cores.ROSA}\tPlotando gráficos de barras agrupadas...{Cores.RESET}")
            graficos_boxplot()
            print(f"\n{Cores.VERDE}\tGráfico(s) plotado(s) com sucesso!{Cores.RESET}")
        elif opcao == "5":
            print(f"\n{Cores.ROSA}\tPlotando gráficos de barras agrupadas...{Cores.RESET}")
            grafico_impacto_variaveis()
            print(f"\n{Cores.VERDE}\tGráfico(s) plotado(s) com sucesso!{Cores.RESET}")
        elif opcao == "0":
            print("Saindo...")
            break
        else:
            print("Opção inválida! Tente novamente.")
        input("\nDigite Enter para continuar...")
        clear()

# Execução do menu
if __name__ == "__main__":
    menu()