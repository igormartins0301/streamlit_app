import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pyti
import plotly.express as px
import plotly.graph_objects as go


st.title('Análise de Rentabilidade em Ações')
st.write('Nesta aplicação, você poderá analisar a rentabilidade de uma ação, utilizando diversas métricas.')

ativo = st.text_input('Digite o código do ativo (exemplo: PETR4.SA)')
percentual = st.text_input(
    'Digite o desconto necessário para comprar a ação (exemplo: 0.015)')


if st.button('Analisar'):
    # Baixar dados do ativo selecionado
    dados_ativo = yf.download(ativo, interval='1d')
    df_ativo = pd.DataFrame(dados_ativo)
    df_ativo['ticker'] = ativo[:-3]  # adiciona uma coluna com o ticker da ação
    df_ativo = df_ativo.reset_index()  # redefine o índice para colunas

    mask = '2013-01-01'
    df_ativo = df_ativo.loc[df_ativo['Date'] > mask]

    df_ativo = df_ativo.sort_values('Date', ascending=True)
    df_ativo['Adj Open'] = df_ativo['Open'] * \
        df_ativo['Adj Close'] / df_ativo['Close']
    df_ativo['Adj High'] = df_ativo['High'] * \
        df_ativo['Adj Close'] / df_ativo['Close']
    df_ativo['Adj Low'] = df_ativo['Low'] * \
        df_ativo['Adj Close'] / df_ativo['Close']

    df_ativo = df_ativo.drop(
        ['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)

    periodo = 1

    df_ativo['fechAnt'] = df_ativo['Adj Close'].shift(1)
    df_ativo['entrada'] = df_ativo['fechAnt'] - \
        (df_ativo['fechAnt'] * float(percentual))

    df_ativo['tem_entrada'] = np.where(
        df_ativo['Adj Low'] < df_ativo['entrada'], 1, 0)
    df_ativo['alvo'] = np.where(df_ativo['tem_entrada'] == 1,
                                (df_ativo['Adj Close'] / df_ativo['entrada'] - 1) * 100, 0)

    df_filtrado = df_ativo.loc[df_ativo['tem_entrada'] == 1]
    df_filtrado['alvo_acumulado'] = df_filtrado['alvo'].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_filtrado['Date'], y=df_filtrado['alvo_acumulado'], mode='lines', name='Modelo sem filtro'))

    fig.update_layout(title='Rentabilidade Acumulada do Ativo')
    st.plotly_chart(fig)

    df_filtrado['max_acumulado'] = df_filtrado['alvo_acumulado'].cummax()
    df_filtrado['drawdown'] = df_filtrado['alvo_acumulado'] - \
        df_filtrado['max_acumulado']

    # Calcular o drawdown máximo
    drawdown_maximo = df_filtrado['drawdown'].min()

    # Localizar o índice do valor mínimo em 'drawdown'
    indice_drawdown_maximo = df_filtrado['drawdown'].idxmin()

    # Acessar a data correspondente ao índice do drawdown máximo
    data_drawdown_maximo = df_filtrado.loc[indice_drawdown_maximo, 'Date']

    st.write("O drawdown máximo de {} ocorreu em {}".format(
        drawdown_maximo.round(2), data_drawdown_maximo))

    # Calcular a taxa de acerto
    resultado = df_filtrado['alvo'] > 0
    taxa_acerto = sum(resultado) / len(df_filtrado['alvo'])

    # Separar resultados positivos e negativos
    resultados_positivos = df_filtrado.loc[df_filtrado['alvo'] > 0, 'alvo']
    resultados_negativos = df_filtrado.loc[df_filtrado['alvo'] <= 0, 'alvo']

    # Calcular médias
    media_positivos = resultados_positivos.mean()
    media_negativos = resultados_negativos.mean()

    # Calcular frequência de resultados positivos e negativos
    freq_positivos = len(resultados_positivos) / len(df_filtrado)
    freq_negativos = len(resultados_negativos) / len(df_filtrado)

    # Calcular expectativa matemática
    expectativa = freq_positivos * media_positivos - \
        freq_negativos * abs(media_negativos)

    # Calcular retorno médio diário
    retorno_medio_diario = df_filtrado['alvo'].mean()

    # Calcular desvio padrão diário
    desvio_padrao_diario = df_filtrado['alvo'].std()

    # Calcular índice de Sharpe
    indice_sharpe = np.sqrt(252) * retorno_medio_diario / desvio_padrao_diario

    st.write('Estatísticas da estratégia')
    st.write('----------------------------------------')
    # Imprimir o resultado
    st.write("Sua taxa de acerto é de {:.2%}".format(taxa_acerto))

    # Imprimir resultados
    st.write("Média de resultados positivos: {:.2f}".format(media_positivos))
    st.write("Média de resultados negativos: {:.2f}".format(media_negativos))

    # Imprimir resultado
    st.write("Expectativa matemática: {:.2f}".format(expectativa))

    # Imprimir resultado
    st.write("Índice de Sharpe: {:.2f}".format(indice_sharpe))
