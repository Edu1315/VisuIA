import streamlit as st
import requests
import json
import pandas as pd
import time

# URL base do backend Flask
FLASK_BASE_URL = "http://127.0.0.1:5000"

st.set_page_config(page_title="Detector de IA em Imagens", layout="centered")
st.title("Detector de Imagens Geradas por IA")
st.markdown(
    "Faça o upload de uma imagem para verificar se ela foi gerada ou alterada por Inteligência Artificial."
)

# Função para Upload e análise de imagem
def upload_image():
    st.header("Upload de Imagem")

    uploaded_files = st.file_uploader(
        "Escolha uma ou mais imagens...",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.write(f"{len(uploaded_files)} imagem(ns) selecionada(s).")

        if len(uploaded_files) >= 1:

            if st.button("Executar Análise"):

                st.write("Iniciando Verificação")
                start_time = time.time()

                files_to_send = [
                    ('imagens', (f.name, f.getvalue()))
                    for f in uploaded_files
                ]

                try:

                    response = requests.post(
                        f"{FLASK_BASE_URL}/analisar_lote",
                        files=files_to_send
                    )

                    if response.status_code == 200:

                        duracao = time.time() - start_time
                        resultados = response.json()

                        st.success(
                            f"Análise concluída em {duracao:.2f} segundos!"
                        )

                        if isinstance(resultados, list):
                            # várias imagens
                            for res in resultados:
                                st.write(res.get("resultado", "Indeterminado"))
                        else:
                            # uma única imagem
                            st.write(resultados.get("resultado", "Indeterminado"))


                        # Se o backend retornar um JSON com "detalhes"
                        lista_resultados = resultados.get(
                            "detalhes",
                            resultados
                        )

                        for arquivo, res in zip(uploaded_files, lista_resultados):
                            st.image(
                                arquivo,
                                caption=arquivo.name,
                                width=400
                            )

                            resultado_texto = res.get("resultado", "Indeterminado")

                            st.write(resultado_texto)

                            st.success("✅ Imagem analisada")
                            st.divider()


                    else:
                        st.error(
                            "Erro no servidor ao processar lote."
                        )

                except Exception as e:
                    st.error(f"Erro: {e}")


# Função para Exibir Histórico de Análises
def display_historico():

    st.header("Histórico de Análises")

    try:

        response = requests.get(
            f"{FLASK_BASE_URL}/historico"
        )

        if response.status_code == 200:

            analises = response.json()

            # após obter analises = response.json()
            if not analises:
                st.info("Nenhuma análise no histórico ainda.")
            else:
                df = pd.DataFrame(analises)
                # manter colunas úteis (se existirem)
                cols = [c for c in ['data', 'arquivo', 'resultado', 'confianca', 'id'] if c in df.columns]
                df = df[cols]

                # renomear para exibição (opcional)
                df = df.rename(columns={
                    'data': 'Data',
                    'arquivo': 'Arquivo',
                    'resultado': 'Resultado',
                    'confianca': 'Confiança',
                    'id': 'ID'
                })

                for index, row in df.iterrows():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        arquivo = row.get('Arquivo') or row.get('arquivo') or "arquivo-desconhecido"
                        resultado = row.get('Resultado') or row.get('resultado') or "Indeterminado"
                        data_iso = row.get('Data') or row.get('data') or ""
                        # formatar ISO para dd/mm/YYYY HH:MM
                        from datetime import datetime, timezone
                        from zoneinfo import ZoneInfo  # disponível a partir do Python 3.9

                        try:
                            dt = datetime.fromisoformat(data_iso)
                            # se vier sem timezone, assume UTC
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            # converte para horário de São Paulo
                            local_dt = dt.astimezone(ZoneInfo("America/Sao_Paulo"))
                            data_fmt = local_dt.strftime("%d/%m/%Y %H:%M")
                        except Exception:
                            data_fmt = data_iso

                        st.markdown(f"*{arquivo}* — {resultado} em {data_fmt}")


                    with col2:
                        if st.button(f"Excluir", key=f"delete_{row.get('ID') or row.get('id')}"):
                            delete_analise(row.get('ID') or row.get('id'))
                            st.rerun()


        else:

            st.error(
                f"Erro ao carregar histórico: "
                f"{response.json().get('error', 'Erro desconhecido')}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Não foi possível conectar ao servidor Flask para carregar o histórico."
        )

    except Exception as e:

        st.error(
            f"Ocorreu um erro inesperado ao carregar o histórico: {e}"
        )


# Função para Excluir Análise
def delete_analise(analise_id):

    try:

        response = requests.delete(
            f"{FLASK_BASE_URL}/historico/{analise_id}"
        )

        if response.status_code == 200:

            st.success(
                "Análise excluída com sucesso!"
            )

        else:

            st.error(
                f"Erro ao excluir análise: "
                f"{response.json().get('error', 'Erro desconhecido')}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Não foi possível conectar ao servidor Flask para excluir a análise."
        )

    except Exception as e:

        st.error(
            f"Ocorreu um erro inesperado ao excluir a análise: {e}"
        )


# Execução das Funções do Frontend
upload_image()
display_historico()