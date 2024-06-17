import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Carregar os dados do arquivo atualizado
file_path = 'Pokemon.xlsx'  # Atualize este caminho conforme necessário
pokemon_df = pd.read_excel(file_path, engine='openpyxl')

# Pergunta: "Quanto a defesa influencia no ataque do Pokémon?"

# Filtrando os dados para remover possíveis valores nulos
filtered_df = pokemon_df[['Name', 'Type 1', 'Attack', 'Defense']].dropna()

# Preparando os dados para regressão linear
X = filtered_df['Defense'].values.reshape(-1, 1)  # Variável independente (Defense)
y = filtered_df['Attack'].values  # Variável dependente (Attack)

# Criando o modelo de regressão linear
regressor = LinearRegression()
regressor.fit(X, y)

# Fazendo previsões
y_pred = regressor.predict(X)

# Adicionando as previsões ao dataframe
filtered_df['Predicted Attack'] = y_pred

# Detalhes da análise
coeficiente = round(regressor.coef_[0], 4)
intercepto = round(regressor.intercept_, 4)
r2 = round(r2_score(y, y_pred), 4)
mae = round(mean_absolute_error(y, y_pred), 4)
mse = round(mean_squared_error(y, y_pred), 4)
rmse = round(mse ** 0.5, 4)

# Cálculos de erro percentual
percentual_erros = ((y - y_pred) / y) * 100
percentual_acertos = 100 - abs(percentual_erros)

media_percentual_erros = round(percentual_erros.mean(), 4)
media_percentual_acertos = round(percentual_acertos.mean(), 4)

# Medida de linearidade por tipo de Pokémon
linearidade_por_tipo = filtered_df.groupby('Type 1').apply(lambda df: round(r2_score(df['Attack'], regressor.predict(df['Defense'].values.reshape(-1, 1))), 4))

# Tipo de Pokémon mais linear e menos linear
tipo_mais_linear = linearidade_por_tipo.idxmax()
linearidade_max = linearidade_por_tipo.max()
tipo_menos_linear = linearidade_por_tipo.idxmin()
linearidade_min = linearidade_por_tipo.min()

# Encontrar o tipo de Pokémon com maior defesa e menor ataque
tipo_maior_defesa = filtered_df.groupby('Type 1')['Defense'].mean().idxmax()
maior_defesa = round(filtered_df.groupby('Type 1')['Defense'].mean().max(), 2)
tipo_menor_ataque = filtered_df.groupby('Type 1')['Attack'].mean().idxmin()
menor_ataque = round(filtered_df.groupby('Type 1')['Attack'].mean().min(), 2)

# Encontrar o tipo de Pokémon com maior ataque e menor defesa
tipo_maior_ataque = filtered_df.groupby('Type 1')['Attack'].mean().idxmax()
maior_ataque = round(filtered_df.groupby('Type 1')['Attack'].mean().max(), 2)
tipo_menor_defesa = filtered_df.groupby('Type 1')['Defense'].mean().idxmin()
menor_defesa = round(filtered_df.groupby('Type 1')['Defense'].mean().min(), 2)

# Layout do Dash
app = dash.Dash(__name__)

# Aba 1: Gráfico interativo
fig = px.scatter(filtered_df, x='Defense', y='Attack', color='Type 1',
                 hover_data=['Name', 'Type 1', 'Attack', 'Defense', 'Predicted Attack'],
                 title='Regressão Linear: Defense vs Attack',
                 labels={"Type 1": "Tipo de Pokémon"})

# Ordenar os dados para a linha de regressão
sorted_data = filtered_df.sort_values('Defense')
fig.add_traces(go.Scatter(x=sorted_data['Defense'], y=regressor.predict(sorted_data['Defense'].values.reshape(-1, 1)),
                          mode='lines', name='Regressão Linear'))

# Layout da aba 1
tab1_content = dcc.Graph(figure=fig)

# Aba 2: Informações adicionais
tab2_content = html.Div([
    html.H2('Informações Adicionais'),
    html.P(f"Coeficiente da regressão (slope): {coeficiente}"),
    html.P(f"Intercepto da regressão: {intercepto}"),
    html.P(f"Coeficiente de determinação (R²): {r2}"),
    html.P(f"Erro absoluto médio (MAE): {mae}"),
    html.P(f"Erro quadrático médio (MSE): {mse}"),
    html.P(f"Raiz do erro quadrático médio (RMSE): {rmse}"),
    html.P(f"Média dos erros percentuais: {media_percentual_erros}%"),
    html.P(f"Média dos acertos percentuais: {media_percentual_acertos}%"),
    html.P(f"Tipo de Pokémon com maior defesa e menor ataque: {tipo_maior_defesa} (Defesa média: {maior_defesa}, Ataque médio: {menor_ataque})"),
    html.P(f"Tipo de Pokémon com maior ataque e menor defesa: {tipo_maior_ataque} (Ataque médio: {maior_ataque}, Defesa média: {menor_defesa})")
])

# Layout do aplicativo Dash
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Gráfico Interativo', children=tab1_content),
        dcc.Tab(label='Informações Adicionais', children=tab2_content)
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
