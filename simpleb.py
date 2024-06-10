import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Carregar os dados do arquivo atualizado
file_path = 'C:/projetosSemGit/AnalaticSimple/Pokemon.xlsx'  # Atualize este caminho conforme necessário
pokemon_df = pd.read_excel(file_path, engine='openpyxl')

# Pergunta genérica: "Quanto o valor de 'Attack' pode influenciar no valor de 'Defense' do Pokémon?"

# Filtrando os dados para remover possíveis valores nulos
filtered_df = pokemon_df[['Name', 'Attack', 'Defense']].dropna()

# Preparando os dados para regressão linear
X = filtered_df['Attack'].values.reshape(-1, 1)  # Variável independente
y = filtered_df['Defense'].values  # Variável dependente

# Criando o modelo de regressão linear
regressor = LinearRegression()
regressor.fit(X, y)

# Fazendo previsões
y_pred = regressor.predict(X)

# Detalhes da análise
coeficiente = regressor.coef_[0]
intercepto = regressor.intercept_
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = mse ** 0.5

# Cálculos de erro percentual
percentual_erros = ((y - y_pred) / y) * 100
percentual_acertos = 100 - abs(percentual_erros)

# Plotando o gráfico com matplotlib
plt.figure(figsize=(10, 6))
plt.scatter(filtered_df['Attack'], filtered_df['Defense'], c='blue', label='Dados')

# Adicionando a linha de regressão linear
plt.plot(X, y_pred, color='black', linewidth=2, label=f'Regressão Linear\nLinha Linear: y = {coeficiente:.2f}x + {intercepto:.2f}')

# Adicionando tooltips usando mplcursors
cursor = mplcursors.cursor(hover=True)
@cursor.connect("add")
def on_add(sel):
    index = sel.target.index
    row = filtered_df.iloc[index]
    sel.annotation.set(text=f"Name: {row['Name']}\nAttack: {row['Attack']}\nDefense: {row['Defense']}\nPredicted: {y_pred[index]:.2f}")

plt.xlabel('Attack')
plt.ylabel('Defense')
plt.title('Regressão Linear: Attack vs Defense')
plt.legend(loc='upper left')

# Adicionando o valor final da linha linear
plt.text(max(filtered_df['Attack']), max(y_pred), f'Final da Linha: {y_pred[-1]:.2f}', fontsize=12, ha='right')

plt.show()

# Exibindo resultados detalhados
print(f"Coeficiente da regressão (slope): {coeficiente:.4f}")
print(f"Intercepto da regressão: {intercepto:.4f}")
print(f"Coeficiente de determinação (R²): {r2:.4f}")
print(f"Erro absoluto médio (MAE): {mae:.4f}")
print(f"Erro quadrático médio (MSE): {mse:.4f}")
print(f"Raiz do erro quadrático médio (RMSE): {rmse:.4f}")
print(f"Média dos erros percentuais: {percentual_erros.mean():.4f}%")
print(f"Média dos acertos percentuais: {percentual_acertos.mean():.4f}%")

# Sumário estatístico dos dados
print("\nSumário estatístico dos dados de Attack")
print(filtered_df['Attack'].describe())
print("\nSumário estatístico dos dados de Defense")
print(filtered_df['Defense'].describe())
