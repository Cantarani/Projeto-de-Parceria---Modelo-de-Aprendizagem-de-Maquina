# importando os módulos que serão usados no decorrer do código
# importing the modules that will be used throughout the code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

""" Criando o dataframe e manipulando os dados """


data = pd.read_csv("suicide_rates_1990-2022.csv")

# Selecionando as colunas de interesse
# Selecting the columns of interest

columns = ["Year", "Sex", "AgeGroup", "Generation", "SuicideCount"]

data = data[columns]

# Remove as linhas com valores nulos na coluna SuicideCount
# Remove rows with null values ​​in the SuicideCount column

data = data.dropna(subset=['SuicideCount'])
data = data[data['SuicideCount'] != 0]

# Agrupar as linhas com base nas informações categóricas e somar as contagens de suicídio
# Group rows based on categorical information and sum suicide counts

data = data.groupby(['Sex', 'AgeGroup', 'Generation', 'Year']).sum().reset_index()

# Remover linhas onde as colunas categóricas com o valor "Unknown"
# Remove rows where categorical columns with value "Unknown"

data = data[(data['Sex'] != 'Unknown') &
            (data['AgeGroup'] != 'Unknown') &
            (data['Generation'] != 'Unknown')]


# Ordenando por AgeGroup e Year na ordem crescente
# Sorting by AgeGroup and Year in ascending order

data = data.sort_values(by=['AgeGroup', 'Year'])
data.reset_index(drop=True, inplace=True)

# Agrupar as contagens de suicídios por AgeGroup e Year
# Group suicide counts by AgeGroup and Year
data_grouped_year = data.groupby(['AgeGroup', 'Year'])['SuicideCount'].sum().reset_index()




""" Gráfico 1 """


# Criar o gráfico de barras
# Create the bar chart
plt.figure(figsize=(8, 4))
for age_group, dados in data_grouped_year.groupby('AgeGroup'):
    plt.plot(dados['Year'], dados['SuicideCount'], label=age_group)
plt.title('Contagens de suicídios por grupo etário ao longo dos anos')
plt.xlabel('Ano')
plt.ylabel('Contagem de suicídios')
plt.legend()
plt.grid(True)

# Armazenar a imagem do gráfico
# Store the chart image
grafico_1 = plt.gcf()



""" Gráfico 2  """



# Filtrar os dados por sexo
# Filter data by gender

df_male = data[data['Sex'] == 'Male']
df_female = data[data['Sex'] == 'Female']

# Agrupar as contagens de suicídios por ano para cada sexo
# Group suicide counts by year for each sex

male_counts = df_male.groupby('Year')['SuicideCount'].sum()
female_counts = df_female.groupby('Year')['SuicideCount'].sum()

# Obter os anos únicos
# Get the unique years
years = data['Year'].unique()

# Criar o gráfico de barras
# Create the bar chart
plt.figure(figsize=(8, 4))

# Adicionar barras para o sexo masculino
# Add slashes for male
plt.bar(years, male_counts, color='blue', alpha=0.5, label='Masculino')

# Adicionar barras para o sexo feminino
# Add slashes for female
plt.bar(years, female_counts, color='orange', alpha=0.5, bottom=male_counts, label='Feminino')

# Configurações do gráfico
# Chart settings
plt.title('Contagem de suicídios por ano e sexo')
plt.xlabel('Ano')
plt.ylabel('Contagem de suicídios')
plt.legend()
plt.grid(True)

# Armazenar a imagem do gráfico
# Store the chart image
grafico_2 = plt.gcf()



""" Gráfico 3  """
# Agrupar as contagens de suicídios por sexo
# Group suicide counts by sex
sex_counts = data.groupby('Sex')['SuicideCount'].sum()

# Definir as cores corretas com transparência
# Set the correct colors with transparency
colors = [(0.8, 0.4, 0.4, 0.5), (0.4, 0.4, 0.8, 0.5)]

# Criar o gráfico de pizza
# Create the pie chart
plt.figure(figsize=(5, 5))
plt.pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%', colors=colors)

# Configurações do gráfico
# Chart settings
plt.title('Porcentagem de suicídios por sexo')

# Armazenar a imagem do gráfico
# Store the chart image
grafico_3 = plt.gcf()


""" Gráfico 4  """
# Criar uma figura e eixos para o gráfico
# Create a figure and axes for the graph
plt.figure(figsize=(6, 4))

# Agrupar as contagens de suicídios por AgeGroup e Sex
# Group suicide counts by AgeGroup and Sex
data_grouped = data.groupby(['AgeGroup', 'Sex'])['SuicideCount'].sum().reset_index()

# Pivotar os dados para ter os sexos como colunas
# Pivot the data to have genders as columns
pivot_data = data_grouped.pivot(index='AgeGroup', columns='Sex', values='SuicideCount')

# Ordenar os grupos etários do mais novo para o mais velho
# Sort age groups from youngest to oldest
pivot_data = pivot_data.reindex(index=pivot_data.index[::-1])

# Cores do gráfico
# Chart colors
colors = [(0.4, 0.4, 0.8, 0.5), (0.8, 0.4, 0.4, 0.5)]

# Plotar o gráfico de barras horizontal
# Plot the horizontal bar chart
sns.barplot(data=pivot_data, y=pivot_data.index, x='Male', color=colors[0], label='Masculino')
sns.barplot(data=pivot_data, y=pivot_data.index, x='Female', color=colors[1], label='Feminino')

# Adicionar título e rótulos dos eixos
# Add axis titles and labels
plt.title('Pirâmide etária de suicídios por sexo')
plt.xlabel('Contagem de suicídios')
plt.ylabel('Faixa Etária')

# Inverter o eixo y para mostrar a faixa etária mais jovem no topo
# Flip the y-axis to show the youngest age group at the top
plt.gca().invert_yaxis()

# Adicionar legenda
# Add caption
plt.legend()

# Ajustar layout
# Adjust layout
plt.tight_layout()

# Armazenar a imagem do gráfico
# Store the chart image
grafico_4 = plt.gcf()


""" Modelo de Machine Learning """
# Remover colunas não necessárias para o agrupamento
dados_agrupamento = data.drop(['Sex', 'AgeGroup', 'Generation'], axis=1)

# Normalizar os dados usando StandardScaler
scaler = StandardScaler()
dados_normalizados = scaler.fit_transform(dados_agrupamento)

# Calcular o WCSS para diferentes valores de k
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(dados_normalizados)
    wcss.append(kmeans.inertia_)

# Treinar o modelo KMeans com 3 clusters
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(dados_normalizados)

# Atribuir os rótulos de cluster aos dados
data['Cluster'] = kmeans.labels_


""" Gráfico 5 """
# Gráfico 5: Gráfico de Dispersão 2D com Cores dos Clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(x='AgeGroup', y='SuicideCount', hue='Cluster', data=data, palette='viridis')
plt.title('Gráfico de Dispersão 2D com Cores dos Clusters')
plt.xlabel('Idade')
plt.ylabel('Contagem de Suicídios')
plt.legend(title='Cluster')
grafico_5 = plt.gcf()

""" Gráfico 6 """
# Gráfico 6: Boxplots por Cluster
plt.figure(figsize=(8, 4))
sns.boxplot(x='Cluster', y='SuicideCount', data=data, palette='Set2')
plt.title('Boxplots de Feature3 por Cluster')
plt.xlabel('Cluster')
plt.ylabel('Feature3')
grafico_6 = plt.gcf()

# Salvando os gráficos em arquivos PNG
grafico_1.savefig('grafico_1.png', dpi=300) # Especificando a resolução (dpi)
grafico_2.savefig('grafico_2.png', dpi=300)
grafico_3.savefig('grafico_3.png', dpi=300)
grafico_4.savefig('grafico_4.png', dpi=300)
grafico_5.savefig('grafico_5.png', dpi=300)  
grafico_6.savefig('grafico_6.png', dpi=300)












