import tempfile
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from loaders import *

ARQUIVOS_VALIDOS = [
    'Site',
    'Youtube',
    'pdf',
    'csv',
    'txt'
]

CONFIG_MODELOS = [
    {
        'Groq': {'modelos': ['llama-3.1-70b-versatile', 'gemma2-9b-it', 'mixtral-8x7b-32768'],
                 'chat': ChatGroq},
        'OpenAI': {'modelos': ['gpt-4o-mini', 'gpt-4o', 'o1-preview', 'o1-mini'],
                   'chat': ChatOpenAI}
    }
]

config_dict = CONFIG_MODELOS[0]
MEMORIA = ConversationBufferMemory()

def extrair_id_youtube(url):
    try:
        parsed_url = urlparse(url)
        if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
            return parse_qs(parsed_url.query).get('v', [None])[0]
        elif parsed_url.hostname == 'youtu.be':
            return parsed_url.path.lstrip('/')
    except:
        return None

def carrega_youtube(url):
    video_id = extrair_id_youtube(url)
    if not video_id:
        return "Erro: URL inválida ou ID não encontrado."
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        texto = " ".join([item['text'] for item in transcript])
        return texto
    except Exception as e:
        return f"Erro ao carregar o vídeo: {e}"

def carrega_arquivos(tipo_arquivo, arquivo):
    if tipo_arquivo == 'Site':
        documento = carregar_site(arquivo)
    if tipo_arquivo == 'Youtube':
        documento = carrega_youtube(arquivo)
    if tipo_arquivo == 'pdf':
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
            temp.write(arquivo.read())
        documento = carrega_pdf(temp.name)
    if tipo_arquivo == 'csv':
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
            temp.write(arquivo.read())
        documento = carrega_csv(temp.name)
    if tipo_arquivo == 'txt':
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
            temp.write(arquivo.read())
        documento = carrega_txt(temp.name)
    return documento

def carregar_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):
    documento = carrega_arquivos(tipo_arquivo, arquivo)
    system_message = f'''Você é um assistente amigável chamado TTData.
    Você possui acesso às seguintes informações vindas 
    de um documento {tipo_arquivo}: 

    ####
    {documento}
    ####

    Utilize as informações fornecidas para basear as suas respostas.

    Sempre que houver $ na sua saída, substitua por S.

    Se a informação do documento for algo como "Just a moment...Enable JavaScript and cookies to continue" 
    sugira ao usuário carregar novamente o TTData!'''
    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])
    chat = config_dict[provedor]['chat'](model=modelo, api_key=api_key)
    chain = template | chat
    st.session_state['chain'] = chain

def pagina_chat():
    st.header('🦾 Bem vindo ao TTData', divider=True)
    chain = st.session_state.get('chain')
    if chain is None:
        st.error('Carregue o TTData')
        st.stop()
    memoria = st.session_state.get('memoria', MEMORIA)
    for mensagem in memoria.buffer_as_messages:
        chat = st.chat_message(mensagem.type)
        chat.markdown(mensagem.content)
    input_usuario = st.chat_input('Fale com TTData')
    if input_usuario:
        chat = st.chat_message('human')
        chat.markdown(input_usuario)
        chat = st.chat_message('ai')
        resposta = chat.write_stream(chain.stream({
            'input': input_usuario,
            'chat_history': memoria.buffer_as_messages
        }))
        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state['memoria'] = memoria

def sidebar():
    tabs = st.tabs(['Upload de arquivos', 'Seleção de modelos'])
    with tabs[0]:
        tipo_arquivo = st.selectbox('Selecione o tipo de arquivo', ARQUIVOS_VALIDOS)
        if tipo_arquivo == 'Site':
            arquivo = st.text_input('Digite a URL do site aqui:')
        if tipo_arquivo == 'Youtube':
            arquivo = st.text_input('Digite a URL do vídeo aqui:')
        if tipo_arquivo in ['pdf', 'csv', 'txt']:
            arquivo = st.file_uploader('Faça o Upload do arquivo aqui:', type=[f'.{tipo_arquivo}'])
    with tabs[1]:
        provedor = st.selectbox('Selecione o provedor do modelo:', config_dict.keys())
        modelo = st.selectbox('Selecione o modelo:', config_dict[provedor]['modelos'])
        api_key = st.text_input(f'Adicione a api key para o provedor {provedor}:', value=st.session_state.get(f'api_key_{provedor}'))
        st.session_state[f'api_key_{provedor}'] = api_key
    if st.button('Inicializar TTData'):
        carregar_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)

def main():
    with st.sidebar:
        sidebar()
    pagina_chat()

if __name__ == '__main__':
    main()