import streamlit as st
import requests
import json
import pandas as pd

# URL base do seu backend Flask
FLASK_BASE_URL = "http://127.0.0.1:5000"

st.set_page_config(page_title="Detector de IA em Imagens", layout="centered" )
st.title("Detector de Imagens Geradas por IA")
st.markdown("Faça o upload de uma imagem para verificar se ela foi gerada ou alterada por Inteligência Artificial.")

# Função para Upload e Análise de Imagem
def upload_image():
    st.header("Upload de Imagem")
    uploaded_file = st.file_uploader("Escolha uma imagem...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Imagem Carregada", use_column_width=True)
        st.write("Analisando imagem...")

        files = {"file": uploaded_file.getvalue()}
        try:
            response = requests.post(f"{FLASK_BASE_URL}/analisar", files=files) #integrção com o backend
            if response.status_code == 200:
                result = response.json()
                st.success("Análise Concluída!")
                st.write(f"**Arquivo:** {result['arquivo']}")
                st.write(f"**Resultado:** {result['resultado']}")
                st.write(f"**Confiança:** {result['confianca']}%")
                st.write(f"**Data:** {result['data']}")
            else:
                st.error(f"Erro na análise: {response.json().get('error', 'Erro desconhecido')}")
        except requests.exceptions.ConnectionError:
            st.error("Não foi possível conectar ao servidor Flask. Certifique-se de que o backend está em execução.")
        except Exception as e:
            st.error(f"Ocorreu um erro inesperado: {e}")

# Função para Exibir Histórico de Análises
def display_historico():
    st.header("Histórico de Análises")
    try:
        response = requests.get(f"{FLASK_BASE_URL}/historico")
        if response.status_code == 200:
            analises = response.json()
            if not analises:
                st.info("Nenhuma análise no histórico ainda.")
            else:
                # Converte para DataFrame para melhor visualização e manipulação
                df = pd.DataFrame(analises)
                df = df[['data', 'arquivo', 'resultado', 'confianca', 'id']]
                df.columns = ['Data', 'Arquivo', 'Resultado', 'Confiança', 'ID'] # Renomeia colunas para exibição

                for index, row in df.iterrows():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{row['Arquivo']}** - {row['Resultado']} ({row['Confiança']}%) em {row['Data']}")
                    with col2:
                        if st.button(f"Excluir", key=f"delete_{row['ID']}"):
                            delete_analise(row['ID'])
                            st.experimental_rerun() # Recarrega a página para atualizar o histórico
        else:
            st.error(f"Erro ao carregar histórico: {response.json().get('error', 'Erro desconhecido')}")
    except requests.exceptions.ConnectionError:
        st.error("Não foi possível conectar ao servidor Flask para carregar o histórico.")
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado ao carregar o histórico: {e}")

# Função para Excluir Análise
def delete_analise(analise_id):
    try:
        response = requests.delete(f"{FLASK_BASE_URL}/historico/{analise_id}")
        if response.status_code == 200:
            st.success("Análise excluída com sucesso!")
        else:
            st.error(f"Erro ao excluir análise: {response.json().get('error', 'Erro desconhecido')}")
    except requests.exceptions.ConnectionError:
        st.error("Não foi possível conectar ao servidor Flask para excluir a análise.")
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado ao excluir a análise: {e}")

# Execução das Funções do Frontend
upload_image()
display_historico()
