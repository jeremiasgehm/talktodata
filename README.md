# TTData - Assistente de Arquivos Inteligente

O **TTData** é um assistente interativo projetado para trabalhar com arquivos diretamente no seu computador. Ele permite que você carregue e interaja com diversos tipos de arquivos e fontes, como sites, vídeos do YouTube, PDFs, arquivos CSV e TXT, integrando-os com modelos de Inteligência Artificial (IA) como **Groq** e **OpenAI**.

## Funcionalidades Principais

### 1. **Carregamento e Processamento de Arquivos**

O TTData suporta o carregamento de diversos tipos de arquivos, cada um com seu método específico de processamento:

- **Site**: Insira a URL de um site para extrair seu conteúdo.
- **YouTube**: Forneça o link de um vídeo e obtenha a transcrição do conteúdo.
- **PDF**: Faça upload de um arquivo PDF para análise.
- **CSV**: Faça upload de um arquivo CSV para processamento e análise.
- **TXT**: Faça upload de um arquivo TXT para leitura.

### 2. **Integração com Modelos de IA**

O TTData pode ser configurado para usar diferentes provedores de modelos de IA:

- **Groq**: Inclui modelos como `llama-3.1-70b-versatile`, `gemma2-9b-it` e `mixtral-8x7b-32768`.
- **OpenAI**: Inclui modelos como `gpt-4o-mini`, `gpt-4o`, `o1-preview`, e `o1-mini`.

O modelo de IA é configurado pelo usuário através da barra lateral, permitindo que o assistente use a IA de acordo com o tipo de arquivo carregado.

### 3. **Interface de Chat Interativa**

A interface de chat permite que você interaja com o TTData de forma simples e direta. O fluxo é o seguinte:

1. Carregue seu arquivo ou insira a URL do site/vídeo.
2. Selecione o provedor de IA e o modelo desejado.
3. Faça perguntas e obtenha respostas baseadas nas informações extraídas do arquivo.

O TTData mantém um histórico de conversas para garantir que as respostas sejam baseadas no contexto e na memória da conversa.

### 4. **Memória de Conversa**

O TTData possui memória integrada para armazenar o histórico de mensagens entre o usuário e o modelo. Isso garante que o assistente consiga fornecer respostas mais contextuais e precisas com o tempo.

### 5. **Funcionalidade de Sidebar**

A barra lateral do TTData oferece opções para:

- **Upload de Arquivos**: Selecione o tipo de arquivo ou cole uma URL para o carregamento.
- **Seleção de Modelos**: Escolha entre diferentes provedores de modelos (Groq ou OpenAI) e selecione o modelo desejado.
- **Chave de API**: Forneça a chave de API necessária para o provedor de IA selecionado.

Após preencher as informações, basta clicar em "Inicializar TTData" para começar a interação com o assistente.

## Como Usar o TTData

### Passo 1: Carregar Arquivos ou Inserir Links

- Escolha o tipo de arquivo (Site, YouTube, PDF, CSV, TXT).
- Se for um site ou vídeo do YouTube, cole a URL.
- Para arquivos locais, faça o upload do arquivo desejado.

### Passo 2: Selecionar o Modelo de IA

Escolha o provedor de IA (Groq ou OpenAI) e selecione o modelo que você deseja usar. Insira sua chave de API, se necessário.

### Passo 3: Interagir com o TTData

Após inicializar o TTData, você pode começar a fazer perguntas sobre o conteúdo carregado. O assistente irá responder com base nas informações fornecidas pelo arquivo.

## Exemplo de Fluxo de Uso

1. O usuário escolhe o tipo de arquivo (por exemplo, um vídeo do YouTube).
2. O TTData extrai a transcrição do vídeo.
3. O usuário escolhe o provedor de IA (exemplo: OpenAI) e o modelo (exemplo: `gpt-4o`).
4. O usuário faz uma pergunta sobre o conteúdo do vídeo.
5. O TTData responde com base na transcrição do vídeo.

## Arquivos Suportados

O TTData oferece suporte a uma variedade de tipos de arquivos:

- **Site**: Carregue conteúdo diretamente de uma URL.
- **YouTube**: Carregue transcrições de vídeos diretamente do YouTube.
- **PDF**: Carregue e analise conteúdo de arquivos PDF.
- **CSV**: Carregue e analise arquivos CSV.
- **TXT**: Carregue e leia conteúdo de arquivos TXT.

## Tecnologias Usadas

- **LangChain**: Para manipulação e interação com modelos de linguagem.
- **Groq e OpenAI**: Provedores de modelos de IA para interação inteligente.
- **YouTube Transcript API**: Para extrair transcrições de vídeos do YouTube.
- **Streamlit**: Para construir a interface de usuário interativa.
- **Tempfile**: Para gerenciar arquivos temporários durante o processamento.

## Conclusão

O TTData é uma ferramenta poderosa para análise e interação com documentos e conteúdos de diferentes fontes. Ele combina inteligência artificial com flexibilidade no processamento de arquivos, oferecendo uma experiência personalizada e interativa.

Experimente o **TTData** hoje e explore como ele pode melhorar sua interação com arquivos e dados!
