from fastapi import FastAPI, HTTPException
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from passlib.context import CryptContext
from pydantic import BaseModel
from jose import jwt
from datetime import datetime, timedelta
import psycopg2
import requests
import os
from dotenv import load_dotenv
import google.generativeai as genai
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import praw
from transformers import pipeline
import pytz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import statistics
from langdetect import detect

load_dotenv()

SECRET_KEY = "w9439568"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = 'inteligencia_competitiva_app'
FACEBOOK_ACCESS_TOKEN = os.getenv("FACEBOOK_ACCESS_TOKEN")

sentiment_pipeline = pipeline(
    "text-classification",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    return_all_scores=True
)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

conn = psycopg2.connect(
    dbname="inteligencia_competitiva",
    user="ic_user",
    password="w9439568",
    host="localhost"
)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS usuarios (
        id SERIAL PRIMARY KEY,
        nome TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        senha_hash TEXT NOT NULL
    )
""")
conn.commit()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# FAISS Index global
faiss_index = None
textos_indexados = []

class UsuarioCreate(BaseModel):
    nome: str
    email: str
    senha: str

class UsuarioLogin(BaseModel):
    email: str
    senha: str

class RequisicaoPesquisa(BaseModel):
    termo: str

def is_portuguese(text):
    try:
        return detect(text) == 'pt'
    except:
        return False

def buscar_noticias(termo):
    url = f"https://newsapi.org/v2/everything?q={termo}&language=pt&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    resp = requests.get(url)
    artigos = resp.json().get("articles", [])[:5]
    return [{"titulo": a["title"], "link": a["url"], "data": a["publishedAt"][:10], "resumo": a["description"]} for a in artigos]

def buscar_site_oficial(termo):
    url = f"https://api.duckduckgo.com/?q={termo}+site+oficial&format=json"
    resp = requests.get(url)
    resultado = resp.json().get("RelatedTopics", [])
    return [r.get("Text") for r in resultado if r.get("Text")]

def buscar_twitter(termo):
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}
    params = {
        "query": f"{termo} lang:pt",  # Filtra por tweets em português
        "max_results": 10,  # Busca mais tweets para aumentar chances de achar em PT
        "tweet.fields": "author_id,text,created_at"
    }
    response = requests.get(url, headers=headers, params=params)
    tweets = response.json().get("data", [])
    
    # Filtra apenas os que são realmente em português (detecção adicional)
    tweets_pt = [tweet for tweet in tweets if is_portuguese(tweet['text'])]
    return [tweet['text'] for tweet in tweets_pt[:5]]

def buscar_reddit(termo):
    submissions = reddit.subreddit('all').search(termo, limit=10)  # Busca mais posts
    posts_pt = []
    
    for submission in submissions:
        try:
            if is_portuguese(submission.title):
                posts_pt.append(submission.title)
                if len(posts_pt) >= 5:  # Limita a 5 posts em português
                    break
        except:
            continue
            
    return posts_pt

def buscar_facebook(termo):
    try:
        url = f"https://graph.facebook.com/v19.0/search"
        params = {
            'q': termo,
            'type': 'page',
            'access_token': FACEBOOK_ACCESS_TOKEN,
            'limit': 10,  # Busca mais resultados
            'locale': 'pt_BR'  # Prioriza conteúdo em português do Brasil
        }
        response = requests.get(url, params=params)
        data = response.json().get('data', [])
        
        # Filtra páginas que parecem ser em português
        resultados = []
        for item in data:
            if is_portuguese(item.get('name', '')):
                resultados.append(f"Facebook Page: {item.get('name')}")
                if len(resultados) >= 3:  # Limita a 3 resultados
                    break
        return resultados
    except Exception as e:
        print(f"Erro ao buscar no Facebook: {e}")
        return []

def buscar_redes_sociais(termo):
    twitter_results = [f"Twitter: {tweet}" for tweet in buscar_twitter(termo)]
    reddit_results = [f"Reddit: {post}" for post in buscar_reddit(termo)]
    facebook_results = buscar_facebook(termo)
    
    return twitter_results + reddit_results + facebook_results
##
def gerar_analise_ia(termo, noticias, redes_sociais, site_oficial, data_pesquisa):
 #   genai.configure(api_key=GEMINI_API_KEY)
 #   model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
    prompt = f"""
Gere um relatório completo de inteligência competitiva para: {termo}

data da pesquisa: {data_pesquisa}

Notícias:
{chr(10).join([n['resumo'] or n['titulo'] for n in noticias])}

Site Oficial:
{chr(10).join(site_oficial)}

Redes Sociais:
{chr(10).join(redes_sociais)}
"""
    resposta = consulta_gemini(prompt)

    ##print("DEBUG - RESPOSTA DA GEMINI:\n", resposta)
    
    return resposta.text if hasattr(resposta, 'text') else str(resposta)
##
@app.post("/principais-produtos")
def principais_produtos(req: RequisicaoPesquisa):
    termo = req.termo.strip()
    prompt = f"""
    Liste os principais produtos ou serviços associados a: {termo}.
    Responda no formato de lista com no máximo 10 itens, apenas os nomes.
    """
    
    resposta = consulta_gemini(prompt)
    
    ##print("DEBUG - RESPOSTA DA GEMINI:\n", resposta)

    produtos = [linha.lstrip("-•*1234567890. ").strip()
                for linha in resposta.splitlines()
                if linha.strip()]
    
    return {
        "data_pesquisa": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "resultado": {
            "produtos": produtos
        }
    }

##
def gerar_analise_swot_produtos(termo):
 #   genai.configure(api_key=GEMINI_API_KEY)
 #   model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

    prompt = f"""

Com base nos seguintes produtos da empresa "{termo}", gere uma análise SWOT (Forças, Fraquezas, Oportunidades, Ameaças):

Produtos:
Formato esperado:
- Forças:
- Fraquezas:
- Oportunidades:
- Ameaças:
    """

    resposta = model.generate_content(prompt)
    return resposta.text if hasattr(resposta, 'text') else str(resposta)


def avaliar_sentimento_para_nota(texto):
    try:
        # Limita o tamanho do texto para o modelo
        texto = texto[:512]
        resultado = sentiment_pipeline(texto)[0]
        
        # Mapeia as avaliações para notas de 1-5
        # O modelo retorna scores para: 1⭐, 2⭐, 3⭐, 4⭐, 5⭐
        scores = {int(item['label'][0]): item['score'] for item in resultado}
        
        # Calcula a nota ponderada
        nota = sum(estrela * score for estrela, score in scores.items())
        return nota
        
    except Exception as e:
        print(f"Erro ao avaliar sentimento: {e}")
        return 3  # Retorna neutro se houver erro

def analisar_aceitacao_publico(textos):
    notas = []
    for texto in textos:
        # Pré-processamento para melhor análise
        texto_limpo = texto.lower().strip()
        
        # Ignora textos muito curtos ou irrelevantes
        if len(texto_limpo.split()) < 3:
            continue
            
        nota = avaliar_sentimento_para_nota(texto_limpo)
        notas.append(nota)
    
    # Se não houver notas válidas, retorna lista vazia
    return notas if notas else []

def gerar_grafico_base64(notas):
    try:
        plt.figure(figsize=(8, 5))
        
        # Categoriza as notas
        categorias = ['Muito Ruim', 'Ruim', 'Neutro', 'Bom', 'Muito Bom']
        valores = [0] * 5
        
        for nota in notas:
            if nota < 1.5:
                valores[0] += 1
            elif nota < 2.5:
                valores[1] += 1
            elif nota < 3.5:
                valores[2] += 1
            elif nota < 4.5:
                valores[3] += 1
            else:
                valores[4] += 1
        
        # Cores baseadas no sentimento
        cores = ['#ff4d4d', '#ff9999', '#ffff99', '#99ff99', '#66b266']
        
        bars = plt.bar(categorias, valores, color=cores)
        
        # Adiciona os valores em cada barra
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.title('Distribuição de Satisfação do Público', pad=20)
        plt.xlabel('Nível de Satisfação')
        plt.ylabel('Quantidade de Menções')
        plt.xticks(rotation=15)
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
        
    except Exception as e:
        print(f"Erro ao gerar gráfico: {e}")
        return ""

def verificar_senha(senha: str, hash: str):
    return pwd_context.verify(senha, hash)

def gerar_hash_senha(senha: str):
    return pwd_context.hash(senha)

def criar_token_acesso(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def buscar_usuario_por_email(email: str):
    cursor.execute("SELECT id, nome, email, senha_hash FROM usuarios WHERE email = %s", (email,))
    row = cursor.fetchone()
    if row:
        return {"id": row[0], "nome": row[1], "email": row[2], "senha_hash": row[3]}
    return None

@app.post("/cadastro")
def cadastrar_usuario(usuario: UsuarioCreate):
    try:
        if buscar_usuario_por_email(usuario.email):
            raise HTTPException(status_code=400, detail="Email já cadastrado")

        hash_senha = gerar_hash_senha(usuario.senha)
        cursor.execute(
            "INSERT INTO usuarios (nome, email, senha_hash) VALUES (%s, %s, %s)",
            (usuario.nome, usuario.email, hash_senha)
        )
        conn.commit()
        return {"mensagem": "Usuário cadastrado com sucesso"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail="Erro interno ao cadastrar usuário")

@app.post("/login")
def login(usuario: UsuarioLogin):
    user = buscar_usuario_por_email(usuario.email)
    if not user or not verificar_senha(usuario.senha, user["senha_hash"]):
        raise HTTPException(status_code=401, detail="Login inválido")
    return {"mensagem": "Login bem-sucedido"}
##

def consulta_gemini(prompt):
    response = model.generate_content(prompt)
    return response.text.strip()

@app.post("/pesquisa")
def pesquisa(dados: RequisicaoPesquisa):
    termo = dados.termo
    tz_brazil = pytz.timezone('America/Sao_Paulo')
    data_pesquisa = datetime.now(tz_brazil).strftime('%d/%m/%Y %H:%M:%S')

    noticias = buscar_noticias(termo)
    site_oficial = buscar_site_oficial(termo)
    redes_sociais = buscar_redes_sociais(termo)

    # Filtra fontes para análise de sentimento
    fontes_para_analise = []
    
    # Adiciona resumos de notícias
    for n in noticias:
        if n.get('resumo'):
            fontes_para_analise.append(n['resumo'])
        elif n.get('titulo'):
            fontes_para_analise.append(n['titulo'])
    
    # Adiciona menções de redes sociais
    fontes_para_analise.extend(redes_sociais)
    
    # Analisa sentimentos
    notas = analisar_aceitacao_publico(fontes_para_analise)
    
    # Calcula métricas
    media = np.mean(notas) if notas else 0
    desvio_padrao = np.std(notas) if notas else 0
    
    # Gera análise da IA
    analise = gerar_analise_ia(termo, noticias, redes_sociais, site_oficial, data_pesquisa)
    
    # Gera gráfico
    grafico_base64 = gerar_grafico_base64(notas)

    return {
        "resultado": {
            "termo": termo,
            "data_pesquisa": data_pesquisa,
            "noticias": noticias,
            "site_oficial": site_oficial,
            "redes_sociais": redes_sociais,
            "analise": analise,
            "aceitacao_publico": {
                "notas": notas,
                "media": round(media, 2),
                "desvio_padrao": round(desvio_padrao, 2),
                "grafico_base64": grafico_base64,
                "total_avaliacoes": len(notas)
            }
        }
    }

@app.post("/principais-produtos")
def principais_produtos(req: RequisicaoPesquisa):
    termo = req.termo.strip()
    prompt = f"""
    Liste os principais produtos ou serviços associados a: {termo}.
    Responda no formato de lista com no máximo 10 itens, apenas os nomes.
    """

    resposta = consulta_gemini(prompt)

    # Extrair itens da lista
    produtos = [linha.lstrip("-•*1234567890. ").strip()
                for linha in resposta.splitlines()
                if linha.strip()]
    
    return {
        "data_pesquisa": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "resultado": {
            "produtos": produtos
        }
    }

@app.post("/analise-swot-dos-produtos")
def analise_swot_dos_produtos(req: RequisicaoPesquisa):
    termo = req.termo
    tz_brazil = pytz.timezone('America/Sao_Paulo')
    data_pesquisa = datetime.now(tz_brazil).strftime('%d/%m/%Y %H:%M:%S')
    
    prompt = f"Faça uma análise SWOT dos principais produtos de '{termo}'. Liste as forças, fraquezas, oportunidades e ameaças."
    resposta = consulta_gemini(prompt)
    
    swot = gerar_analise_swot_produtos(resposta)
    
    return {
        "data_pesquisa": data_pesquisa,
        "resultado": {
            "swot": swot
        }
    }

##
@app.post("/similaridade")
def similaridade(data: dict):
    global faiss_index, textos_indexados
    termo = data.get("termo")

    if not textos_indexados:
        return {"mensagem": "Banco vetorial vazio, realize pesquisas primeiro."}

    embedding_termo = embedding_model.encode([termo])
    D, I = faiss_index.search(np.array(embedding_termo), k=5)
    similares = [textos_indexados[i] for i in I[0] if i != -1]

    return {"termo_consulta": termo, "resultados_similares": similares}

def indexar_textos_para_faiss(textos):
    global faiss_index, textos_indexados

    embeddings = embedding_model.encode(textos)
    d = embeddings.shape[1]

    faiss_index = faiss.IndexFlatL2(d)
    faiss_index.add(np.array(embeddings))
    textos_indexados = textos

