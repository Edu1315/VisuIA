import streamlit as st
import requests
import json
import pandas as pd
import time

# URL base do backend Flask
FLASK_BASE_URL = "http://127.0.0.1:5000"

st.set_page_config(page_title="Detector de IA em Imagens", layout="centered" )
st.title("Detector de Imagens Geradas por IA")
st.markdown("Faça o upload de uma imagem para verificar se ela foi gerada ou alterada por Inteligência Artificial.")

# Função para Upload e análise de imagem
def upload_image():
    st.header("Upload de Imagem")
    
    # Agora aceita múltiplos arquivos
    uploaded_files = st.file_uploader(
        "Escolha uma ou mais imagens...", 
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True 
    )

    if uploaded_files:
        st.write(f"{len(uploaded_files)} imagem(ns) selecionada(s).")     
        
        # Botão novo (Paralelismo)
        if len(uploaded_files) >= 1:
            if st.button("Executar Análise"):
                st.write("Iniciando processamento paralelo...")
                start_time = time.time()
                
                # Prepara o lote de arquivos
                files_to_send = [('imagens', (f.name, f.getvalue())) for f in uploaded_files]
                
                try:
                    response = requests.post(f"{FLASK_BASE_URL}/analisar_lote", files=files_to_send)
                    
                    if response.status_code == 200:
                        duracao = time.time() - start_time
                        resultados = response.json()
                        
                        st.success(f"Concluído em {duracao:.2f} segundos!")
                        for r in resultados:
                            st.write(f"✅ {r}")
                        
                        st.info(f"Explicação: Se fosse sem paralelismo, levaria o dobro do tempo!")
                    else:
                        st.error("Erro no servidor ao processar lote.")
                except Exception as e:
                    st.error(f"Erro: {e}")

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
                            st.rerun() # Recarrega a página para atualizar o histórico
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