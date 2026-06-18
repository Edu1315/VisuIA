# VisuIA

## Sobre o Projeto

O VisuIA é uma aplicação web desenvolvida para identificar a probabilidade de uma imagem ter sido gerada por Inteligência Artificial.

A solução utiliza um modelo de Deep Learning baseado na arquitetura MobileNetV2 para analisar imagens enviadas pelos usuários, retornando o resultado da classificação juntamente com o nível de confiança da predição.

O sistema é composto por um frontend desenvolvido em Streamlit, uma API REST construída com Flask, um modelo de Inteligência Artificial treinado em TensorFlow/Keras e um banco de dados MongoDB Atlas para armazenamento do histórico das análises.

---

## Objetivo

Permitir que usuários realizem o upload de imagens e obtenham uma estimativa sobre a possibilidade de essas imagens terem sido geradas ou manipuladas por Inteligência Artificial.

Além disso, o sistema mantém um histórico das análises realizadas para consulta posterior.

---

## Tecnologias Utilizadas

### Backend

* Python
* Flask
* Waitress

### Frontend

* Streamlit

### Inteligência Artificial

* TensorFlow
* Keras
* MobileNetV2

### Banco de Dados

* MongoDB Atlas
* Flask-PyMongo

### Bibliotecas Auxiliares

* Pandas
* Pillow (PIL)
* NumPy
* python-dotenv
* ThreadPoolExecutor

---

## Arquitetura da Aplicação

```text
Usuário
   ↓
Frontend (Streamlit)
   ↓
API REST (Flask)
   ↓
Modelo de IA (MobileNetV2)
   ↓
MongoDB Atlas
```

---

## Funcionalidades

### Upload de Imagens

Permite o envio de uma ou múltiplas imagens para análise.

### Análise Individual

Processamento de uma única imagem utilizando o modelo de Inteligência Artificial.

### Análise em Lote

Processamento simultâneo de múltiplas imagens utilizando execução paralela através de threads.

### Classificação por IA

O modelo analisa a imagem e retorna a probabilidade de ter sido gerada por Inteligência Artificial.

### Nível de Confiança

Exibição do percentual de confiança associado à predição realizada pelo modelo.

### Histórico de Análises

Armazenamento de todas as análises realizadas no MongoDB Atlas.

### Consulta de Histórico

Visualização das análises previamente executadas.

### Exclusão de Registros

Remoção de análises armazenadas no histórico.

---

## Estrutura do Projeto

```text
VisuIA/
│
├── backend/
│   ├── app.py
│   ├── inferencia.py
│   ├── testes.py
│   │
│   ├── database/
│   │   └── mongo.py
│   │
│   └── services/
│       └── detector_service.py
│
├── frontend/
│   └── streamlit_app.py
│
├── model/
│   └── mobilenet_v2.h5
│
├── uploads/
│
├── scripts/
│
├── .env
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Endpoints da API

### Status da Aplicação

```http
GET /status
```

Retorna informações sobre o estado da API.

---

### Análise de Imagem

```http
POST /analisar
```

Recebe uma imagem e retorna o resultado da classificação.

---

### Análise em Lote

```http
POST /analisar_lote
```

Recebe múltiplas imagens e processa todas em paralelo.

---

### Consultar Histórico

```http
GET /historico
```

Retorna as análises armazenadas no banco de dados.

---

### Consultar Análise Específica

```http
GET /historico/{id}
```

Retorna os dados de uma análise específica.

---

### Excluir Análise

```http
DELETE /historico/{id}
```

Remove uma análise do histórico.

---

## Configuração do Ambiente

### Clonar o Repositório

```bash
git clone <url-do-repositorio>
cd VisuIA
```

### Criar Ambiente Virtual

```bash
python -m venv .venv
```

### Ativar Ambiente Virtual

Windows:

```bash
.venv\Scripts\activate
```

### Instalar Dependências

```bash
pip install -r requirements.txt
```

---

## Configuração do MongoDB

Criar um arquivo `.env` na raiz do projeto contendo:

```env
MONGO_URI=sua_string_de_conexao_mongodb
```

---

## Executando o Backend

```bash
python backend/app.py
```

A API será iniciada em:

```text
http://127.0.0.1:5000
```

---

## Executando o Frontend

```bash
streamlit run frontend/streamlit_app.py
```

O Streamlit disponibilizará uma URL local para acesso da interface.

---

## Fluxo de Funcionamento

1. O usuário realiza o upload de uma ou mais imagens.
2. O frontend envia os arquivos para a API Flask.
3. O backend processa as imagens.
4. O modelo MobileNetV2 realiza a inferência.
5. O resultado é retornado para o frontend.
6. As informações da análise são armazenadas no MongoDB Atlas.
7. O histórico pode ser consultado ou excluído posteriormente.

---

## Inteligência Artificial Utilizada

O sistema utiliza um modelo de Deep Learning salvo no arquivo:

```text
model/mobilenet_v2.h5
```

O modelo é carregado pelo TensorFlow/Keras durante a inicialização da aplicação e utilizado para realizar a classificação das imagens enviadas pelos usuários.

---

## Projeto Acadêmico

Este projeto foi desenvolvido como atividade acadêmica do curso de Análise e Desenvolvimento de Sistemas, aplicando conceitos de:

* Inteligência Artificial
* Visão Computacional
* APIs REST
* Banco de Dados NoSQL
* Desenvolvimento Web com Python
* Processamento de Imagens
* Arquitetura Cliente-Servidor

---

## Equipe

Projeto desenvolvido pela equipe VisuIA.
