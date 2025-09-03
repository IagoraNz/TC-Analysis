# Bibliotecas necessárias --------------------------------------------------------------------------------------------------------------------- #

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

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

def extrair_titulo(nome):
    """Extrai títulos dos nomes dos passageiros"""
    if pd.isna(nome):
        return 'Unknown'
    titulo = nome.split(',')[1].split('.')[0].strip()
    # Agrupa títulos similares
    if titulo in ['Mr']:
        return 'Mr'
    elif titulo in ['Miss', 'Ms', 'Mlle']:
        return 'Miss'
    elif titulo in ['Mrs', 'Mme']:
        return 'Mrs'
    elif titulo in ['Master']:
        return 'Master'
    else:
        return 'Other'

def extrair_tamanho_familia(dados):
    """Extrai o tamanho da família"""
    return dados['SibSp'] + dados['Parch'] + 1

def preprocessar_dados(dados):
    """Função para pré-processar os dados do Titanic"""
    dados = dados.copy()
    
    # Feature Engineering
    dados['Title'] = dados['Name'].apply(extrair_titulo)
    dados['FamilySize'] = extrair_tamanho_familia(dados)
    dados['IsAlone'] = (dados['FamilySize'] == 1).astype(int)
    
    # Tratar valores ausentes
    # Idade: preencher com mediana por classe e sexo
    for classe in dados['Pclass'].unique():
        for sexo in dados['Sex'].unique():
            mask = (dados['Pclass'] == classe) & (dados['Sex'] == sexo)
            mediana_idade = dados[mask]['Age'].median()
            dados.loc[mask & dados['Age'].isna(), 'Age'] = mediana_idade
    
    # Fare: preencher com mediana por classe
    for classe in dados['Pclass'].unique():
        mediana_fare = dados[dados['Pclass'] == classe]['Fare'].median()
        dados.loc[(dados['Pclass'] == classe) & dados['Fare'].isna(), 'Fare'] = mediana_fare
    
    # Embarked: preencher com moda
    moda_embarked = dados['Embarked'].mode()[0] if not dados['Embarked'].mode().empty else 'S'
    dados['Embarked'].fillna(moda_embarked, inplace=True)
    
    # Cabin: criar variável binária indicando presença de cabine
    dados['HasCabin'] = (~dados['Cabin'].isna()).astype(int)
    
    # Tratar outliers nas variáveis numéricas
    colunas_numericas = ['Age', 'Fare']
    for coluna in colunas_numericas:
        if dados[coluna].notna().sum() > 0:
            tratar_outliers(dados, coluna)
    
    return dados

CAMINHO_ARQUIVO = 'Titanic-Dataset.csv' 

# Carregamento dos dados
print(f"{Cores.AZUL}Carregando dados do arquivo: {CAMINHO_ARQUIVO}{Cores.RESET}")
dados_csv = pd.read_csv(CAMINHO_ARQUIVO)

# Pré-processar os dados
dados_csv = preprocessar_dados(dados_csv)

# ------------------------------------------------------------------------------------------------------------------------------ #
# Funções para cada opção do menu

def painel_estatistica():
    print_quadro("Informações relativas à base de dados do Titanic")
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
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    sns.histplot(dados_csv['Age'], bins=20, kde=False, color='skyblue')
    plt.title("Distribuição da Idade dos Passageiros")
    plt.xlabel("Idade")
    plt.ylabel("Frequência")
    
    plt.subplot(2, 3, 2)
    sns.histplot(dados_csv['Fare'], bins=20, kde=False, color='lightcoral')
    plt.title("Distribuição das Tarifas")
    plt.xlabel("Tarifa")
    plt.ylabel("Frequência")
    
    plt.subplot(2, 3, 3)
    sns.histplot(dados_csv['FamilySize'], bins=10, kde=False, color='lightgreen')
    plt.title("Distribuição do Tamanho da Família")
    plt.xlabel("Tamanho da Família")
    plt.ylabel("Frequência")
    
    plt.subplot(2, 3, 4)
    sns.countplot(data=dados_csv, x='Pclass', color='gold')
    plt.title("Distribuição por Classe")
    plt.xlabel("Classe")
    plt.ylabel("Número de Passageiros")
    
    plt.subplot(2, 3, 5)
    sns.countplot(data=dados_csv, x='Embarked', color='mediumpurple')
    plt.title("Distribuição por Porto de Embarque")
    plt.xlabel("Porto de Embarque")
    plt.ylabel("Número de Passageiros")
    
    plt.subplot(2, 3, 6)
    sns.countplot(data=dados_csv, x='Title', color='orange')
    plt.title("Distribuição por Título")
    plt.xlabel("Título")
    plt.ylabel("Número de Passageiros")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def graficos_barras_agrupadas():
    plt.figure(figsize=(15, 10))
    palette_azul = ["#FF6B6B", "#4ECDC4"]
    
    plt.subplot(2, 3, 1)
    sns.countplot(data=dados_csv, x='Sex', hue='Survived', palette=palette_azul)
    plt.title("Sobrevivência por Sexo")
    plt.legend(['Não Sobreviveu', 'Sobreviveu'])
    
    plt.subplot(2, 3, 2)
    sns.countplot(data=dados_csv, x='Pclass', hue='Survived', palette=palette_azul)
    plt.title("Sobrevivência por Classe")
    plt.legend(['Não Sobreviveu', 'Sobreviveu'])
    
    plt.subplot(2, 3, 3)
    sns.countplot(data=dados_csv, x='Embarked', hue='Survived', palette=palette_azul)
    plt.title("Sobrevivência por Porto de Embarque")
    plt.legend(['Não Sobreviveu', 'Sobreviveu'])
    
    plt.subplot(2, 3, 4)
    sns.countplot(data=dados_csv, x='Title', hue='Survived', palette=palette_azul)
    plt.title("Sobrevivência por Título")
    plt.xticks(rotation=45)
    plt.legend(['Não Sobreviveu', 'Sobreviveu'])
    
    plt.subplot(2, 3, 5)
    sns.countplot(data=dados_csv, x='IsAlone', hue='Survived', palette=palette_azul)
    plt.title("Sobrevivência: Sozinho vs Com Família")
    plt.xlabel("0: Com Família, 1: Sozinho")
    plt.legend(['Não Sobreviveu', 'Sobreviveu'])
    
    plt.subplot(2, 3, 6)
    sns.countplot(data=dados_csv, x='HasCabin', hue='Survived', palette=palette_azul)
    plt.title("Sobrevivência: Possuía Cabine")
    plt.xlabel("0: Sem Cabine, 1: Com Cabine")
    plt.legend(['Não Sobreviveu', 'Sobreviveu'])
    
    plt.tight_layout()
    plt.show()

def graficos_boxplot():
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.boxplot(x='Survived', y='Age', data=dados_csv, showfliers=False)
    plt.title("Boxplot de Idade por Sobrevivência")
    plt.xlabel("Sobreviveu (0: Não, 1: Sim)")
    
    plt.subplot(1, 3, 2)
    sns.boxplot(x='Survived', y='Fare', data=dados_csv, showfliers=False)
    plt.title("Boxplot de Tarifa por Sobrevivência")
    plt.xlabel("Sobreviveu (0: Não, 1: Sim)")
    
    plt.subplot(1, 3, 3)
    sns.boxplot(x='Pclass', y='Fare', data=dados_csv, showfliers=False)
    plt.title("Boxplot de Tarifa por Classe")
    plt.xlabel("Classe")
    
    plt.tight_layout()
    plt.show()

def analise_padroes_sobrevivencia():
    print_quadro("Análise de Padrões de Sobrevivência")
    
    # Taxa de sobrevivência geral
    taxa_sobrevivencia_geral = dados_csv['Survived'].mean()
    print(f"\n{Cores.AZUL}Taxa de sobrevivência geral: {taxa_sobrevivencia_geral:.2%}{Cores.RESET}")
    
    # Taxa de sobrevivência por sexo
    print(f"\n{Cores.AMARELO}Taxa de sobrevivência por sexo:{Cores.RESET}")
    sobrev_sexo = dados_csv.groupby('Sex')['Survived'].agg(['count', 'sum', 'mean'])
    sobrev_sexo.columns = ['Total', 'Sobreviveram', 'Taxa']
    print(sobrev_sexo)
    
    # Taxa de sobrevivência por classe
    print(f"\n{Cores.AMARELO}Taxa de sobrevivência por classe:{Cores.RESET}")
    sobrev_classe = dados_csv.groupby('Pclass')['Survived'].agg(['count', 'sum', 'mean'])
    sobrev_classe.columns = ['Total', 'Sobreviveram', 'Taxa']
    print(sobrev_classe)
    
    # Taxa de sobrevivência por faixa etária
    dados_csv['Faixa_Idade'] = pd.cut(dados_csv['Age'], bins=[0, 18, 35, 60, 100], 
                                     labels=['Criança/Adolescente', 'Jovem Adulto', 'Adulto', 'Idoso'])
    print(f"\n{Cores.AMARELO}Taxa de sobrevivência por faixa etária:{Cores.RESET}")
    sobrev_idade = dados_csv.groupby('Faixa_Idade')['Survived'].agg(['count', 'sum', 'mean'])
    sobrev_idade.columns = ['Total', 'Sobreviveram', 'Taxa']
    print(sobrev_idade)
    
    # Análise de correlação
    print(f"\n{Cores.AMARELO}Matriz de correlação das variáveis numéricas:{Cores.RESET}")
    colunas_numericas = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone', 'HasCabin']
    correlacoes = dados_csv[colunas_numericas].corr()['Survived'].sort_values(ascending=False)
    print(correlacoes)

def grafico_impacto_variaveis():
    # Preparar dados para regressão logística
    dados_modelo = dados_csv.copy()
    
    # Codificar variáveis categóricas
    le_sex = LabelEncoder()
    dados_modelo['Sex_encoded'] = le_sex.fit_transform(dados_modelo['Sex'])
    
    le_embarked = LabelEncoder()
    dados_modelo['Embarked_encoded'] = le_embarked.fit_transform(dados_modelo['Embarked'])
    
    # Selecionar features
    features = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone', 'HasCabin']
    X = dados_modelo[features].copy()
    y = dados_modelo['Survived'].copy()
    
    # Normalizar features
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Adicionar constante
    X_scaled = sm.add_constant(X_scaled)
    
    # Ajustar modelo
    modelo = sm.Logit(y, X_scaled).fit()
    print("\n")
    print_quadro("Sumário do Modelo de Regressão Logística")
    print(modelo.summary())
    
    # Gráfico de coeficientes
    coeficientes = modelo.params[1:]  # Excluir constante
    labels = ['Classe', 'Sexo', 'Idade', 'Irmãos/Cônjuges', 'Pais/Filhos', 'Tarifa', 'Tam. Família', 'Sozinho', 'Tem Cabine']
    
    plt.figure(figsize=(12, 6))
    cores = ['red' if coef < 0 else 'green' for coef in coeficientes]
    bars = plt.bar(labels, coeficientes, color=cores, alpha=0.7)
    plt.title("Impacto das Variáveis na Sobrevivência (Coeficientes Logit)")
    plt.xlabel("Variáveis")
    plt.ylabel("Coeficiente")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Gráfico da matriz de correlação
    plt.figure(figsize=(10, 8))
    colunas_corr = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone', 'HasCabin']
    matriz_corr = dados_csv[colunas_corr].corr()
    sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', center=0, square=True)
    plt.title("Matriz de Correlação das Variáveis")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------------------------------------------------------ #
# Menu principal

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def menu():
    while True:
        print_quadro("ANÁLISE DO DATASET TITANIC", Cores.AZUL)
        print("1. Visualizar painel de estatísticas")
        print("2. Plotar gráficos de histograma")
        print("3. Plotar gráficos de barras agrupadas")
        print("4. Plotar boxplots")
        print("5. Plotar gráficos de dispersão")
        print("6. Análise de padrões de sobrevivência")
        print("7. Gráfico de impacto das variáveis")
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
            print(f"\n{Cores.ROSA}\tPlotando boxplots...{Cores.RESET}")
            graficos_boxplot()
            print(f"\n{Cores.VERDE}\tGráfico(s) plotado(s) com sucesso!{Cores.RESET}")
        elif opcao == "5":
            print(f"\n{Cores.ROSA}\tAnalisando padrões de sobrevivência...{Cores.RESET}")
            analise_padroes_sobrevivencia()
            print(f"\n{Cores.VERDE}\tAnálise concluída com sucesso!{Cores.RESET}")
        elif opcao == "6":
            print(f"\n{Cores.ROSA}\tAnalisando impacto das variáveis...{Cores.RESET}")
            grafico_impacto_variaveis()
            print(f"\n{Cores.VERDE}\tAnálise concluída com sucesso!{Cores.RESET}")
        elif opcao == "0":
            print("Saindo...")
            break
        else:
            print("Opção inválida! Tente novamente.")
        input("\nDigite Enter para continuar...")
        clear()

# Execução do menu
if __name__ == "__main__":
    print_quadro("ATIVIDADE PRÁTICA I - ANÁLISE DO DATASET TITANIC", Cores.VERDE)
    print("Desenvolvido seguindo os requisitos da disciplina Tópicos em Computação")
    print("Dataset: Titanic - Análise de Sobrevivência dos Passageiros")
    print(f"\nTentando carregar arquivo: {CAMINHO_ARQUIVO}")
    
    try:
        # Pré-processar os dados
        dados_csv = preprocessar_dados(dados_csv)
        print(f"{Cores.VERDE}Dados carregados e pré-processados com sucesso!{Cores.RESET}")
        print(f"Total de registros: {len(dados_csv)}")
        menu()
    except FileNotFoundError:
        print(f"{Cores.ROSA}❌ ERRO: Arquivo não encontrado!{Cores.RESET}")
        print(f"{Cores.AMARELO}Verifique se o arquivo '{CAMINHO_ARQUIVO}' existe e está no local correto.{Cores.RESET}")
        print(f"{Cores.AZUL}Para corrigir: Altere a variável CAMINHO_ARQUIVO no início do código.{Cores.RESET}")
    except Exception as e:
        print(f"{Cores.ROSA}❌ ERRO ao carregar o arquivo: {e}{Cores.RESET}")
        print(f"{Cores.AMARELO}Verifique se o arquivo está no formato CSV correto.{Cores.RESET}")