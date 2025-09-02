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
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Histograma da idade
    sns.histplot(dados_csv['age'], bins=20, kde=False, ax=axs[0], color='#87CEFA')  # azul pastel
    axs[0].set_title("Histograma da Idade")
    axs[0].set_xlabel("Idade")
    axs[0].set_ylabel("Número de pessoas")

    # Histograma de horas por semana
    sns.histplot(dados_csv['hours-per-week'], bins=20, kde=False, ax=axs[1], color='#9370DB')  # roxo um pouco mais saturado
    axs[1].set_title("Distribuição de Horas/Semana")
    axs[1].set_xlabel("Horas por Semana")
    axs[1].set_ylabel("Número de Pessoas")

    # Histograma da escolaridade
    sns.histplot(dados_csv['educational-num'], bins=15, kde=False, ax=axs[2], color='#90EE90')  # verde pastel
    axs[2].set_title("Distribuição da Escolaridade")
    axs[2].set_xlabel("Anos de Escolaridade")
    axs[2].set_ylabel("Número de Pessoas")

    plt.tight_layout()
    plt.show()

def graficos_barras_agrupadas():
    palette_contraste = ["#9370DB", "#90EE90"]  # roxo e verde pastel

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Contagem por sexo e renda
    sns.countplot(data=dados_csv, x='gender', hue='income', palette=palette_contraste, ax=axs[0])
    axs[0].set_title("Contagem por Sexo e Renda")
    axs[0].set_xlabel("Sexo")
    axs[0].set_ylabel("Número de Pessoas")
    axs[0].legend(title="Renda")  # legenda ajustada

    # Contagem por raça e renda
    sns.countplot(data=dados_csv, x='race', hue='income', palette=palette_contraste, ax=axs[1])
    axs[1].set_title("Renda por Raça")
    axs[1].set_xlabel("Raça")
    axs[1].set_ylabel("Número de Pessoas")
    axs[1].legend(title="Renda")  # legenda ajustada

    plt.tight_layout()
    plt.show()

def graficos_boxplot():
    _, axs = plt.subplots(1, 2, figsize=(14, 6))

    palette_azul_pastel = {"<=50K": "#87CEFA", ">50K": "#87CEFA"}

    sns.boxplot(
        x='income',
        y='hours-per-week',
        data=dados_csv,
        showfliers=False,
        ax=axs[0],
        hue='income',
        palette=palette_azul_pastel,
        dodge=False
    )
    axs[0].set_title("Horas por semana vs Renda")
    axs[0].set_xlabel("Renda")
    axs[0].set_ylabel("Horas por Semana")
    if axs[0].get_legend() is not None:
        axs[0].get_legend().remove()

    sns.boxplot(
        x='income',
        y='age',
        data=dados_csv,
        showfliers=False,
        ax=axs[1],
        hue='income',
        palette=palette_azul_pastel,
        dodge=False
    )
    axs[1].set_title("Idade vs Renda")
    axs[1].set_xlabel("Renda")
    axs[1].set_ylabel("Idade")
    if axs[1].get_legend() is not None:
        axs[1].get_legend().remove()

    plt.tight_layout()
    plt.show()

def graficos_linhas():
    renda_classes = dados_csv['income'].unique()
    cores_renda = {"<=50K": "#1f77b4", ">50K": "#ff7f0e"}  # cores por renda

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))  # 2x2 subplots

    # 1. Distribuição de Pessoas por Faixa de Idade e Renda
    idade_bins = range(int(dados_csv['age'].min()), int(dados_csv['age'].max()) + 5, 5)
    for income in renda_classes:
        subset = dados_csv[dados_csv['income'] == income]
        counts, bins = np.histogram(subset['age'], bins=idade_bins)
        axs[0, 0].plot(bins[:-1], counts, marker='o', label=f'{income}', color=cores_renda[income])

    axs[0, 0].set_title("Distribuição de Pessoas por Faixa de Idade e Renda")
    axs[0, 0].set_xlabel("Idade")
    axs[0, 0].set_ylabel("Número de Pessoas")
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)
    axs[0, 0].legend(loc='upper right')

    # 2. Distribuição de Renda por Nível de Escolaridade
    escolaridade_map = {
        1: "Pré-escola", 2: "1º-4º", 3: "5º-6º", 4: "7º-8º",
        5: "9º", 6: "10º", 7: "11º", 8: "12º", 9: "Ensino Médio",
        10: "Algum Ensino Superior", 11: "Curso Técnico/Associado", 
        12: "Graduação", 13: "Mestrado", 14: "Escola Profissional", 15: "Doutorado"
    }
    escolaridade_ordem = list(escolaridade_map.keys())

    # Plot das linhas por renda com label
    for income in renda_classes:
        subset = dados_csv[dados_csv['income'] == income]
        counts = subset.groupby('educational-num').size().reindex(escolaridade_ordem, fill_value=0)
        axs[0, 1].plot(escolaridade_ordem, counts, marker='o', color=cores_renda[income], label=income)

    axs[0, 1].set_title("Distribuição de Renda por Nível de Escolaridade")
    axs[0, 1].set_xlabel("Código da Escolaridade")
    axs[0, 1].set_ylabel("Número de Pessoas")
    axs[0, 1].set_xticks(escolaridade_ordem)
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)

    # Legenda de renda (cores das linhas) - canto superior direito
    legenda_renda = axs[0, 1].legend(loc='upper right', fontsize=9)
    axs[0, 1].add_artist(legenda_renda)  # mantém a legenda ativa

    # Legenda de escolaridade (número → nome) - canto superior esquerdo
    from matplotlib.lines import Line2D
    legend_escolaridade = [Line2D([0], [0], color='black', lw=0, label=f"{k}: {v}") 
                           for k, v in escolaridade_map.items()]
    axs[0, 1].legend(handles=legend_escolaridade, loc='upper left', fontsize=8, frameon=True)

    # 3. Distribuição de Renda por Faixa Etária (grupos de 10 anos)
    idade_bins_10 = range(0, 101, 10)
    idade_labels = [f"{i}-{i+9}" for i in idade_bins_10[:-1]]
    for income in renda_classes:
        subset = dados_csv[dados_csv['income'] == income]
        counts, bins = np.histogram(subset['age'], bins=idade_bins_10)
        axs[1, 0].plot(idade_labels, counts, marker='o', label=f'{income}', color=cores_renda[income])

    axs[1, 0].set_title("Distribuição de Renda por Faixa Etária")
    axs[1, 0].set_xlabel("Faixa Etária")
    axs[1, 0].set_ylabel("Número de Pessoas")
    axs[1, 0].tick_params(axis='x', rotation=45)
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)
    axs[1, 0].legend(loc='upper right')

    # 4. Distribuição de Renda por Tipo de Relacionamento
    relacionamentos = sorted(dados_csv['relationship'].unique())
    for income in renda_classes:
        subset = dados_csv[dados_csv['income'] == income]
        counts = subset.groupby('relationship').size().reindex(relacionamentos, fill_value=0)
        axs[1, 1].plot(relacionamentos, counts, marker='o', label=f'{income}', color=cores_renda[income])

    axs[1, 1].set_title("Distribuição de Renda por Tipo de Relacionamento")
    axs[1, 1].set_xlabel("Relacionamento")
    axs[1, 1].set_ylabel("Número de Pessoas")
    axs[1, 1].grid(True, linestyle='--', alpha=0.5)
    axs[1, 1].legend(loc='upper right')

    plt.tight_layout()
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
        print("5. Gráfico de barra (impacto das variáveis)")
        print("6. Gráficos de linhas")
        print("0. Sair")

        opcao = input("Escolha uma opção: ")

        if opcao == "1":
            painel_estatistica()
        elif opcao == "2":
            print(f"\n{Cores.ROSA}\tPlotando histogramas...{Cores.RESET}")
            graficos_histograma()
        elif opcao == "3":
            print(f"\n{Cores.ROSA}\tPlotando gráficos de barras agrupadas...{Cores.RESET}")
            graficos_barras_agrupadas()
        elif opcao == "4":
            print(f"\n{Cores.ROSA}\tPlotando boxplots...{Cores.RESET}")
            graficos_boxplot()
        elif opcao == "5":
            print(f"\n{Cores.ROSA}\tPlotando gráfico de impacto das variáveis...{Cores.RESET}")
            grafico_impacto_variaveis()
        elif opcao == "6":
            print(f"\n{Cores.ROSA}\tPlotando gráficos de linhas...{Cores.RESET}")
            graficos_linhas()
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