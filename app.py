import os
import re
import openai
import streamlit as st
from langchain.llms import OpenAIChat
from langchain.chains import VectorDBQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer in Markdown format:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


@st.cache_data
def split_youtube(url, chunk_chars=4000, overlap=400):
    """
    Pre-process YouTube transcript into chunks
    """
    st.info("Reading and splitting YouTube transcript ...")
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_chars, chunk_overlap=overlap)
    splits = loader.load_and_split(text_splitter=splitter)
    return splits


# @st.cache_resource
def create_ix(splits):
    """ 
    Create vector DB index of PDF
    """
    st.info("`Building index ...`")
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(splits, embeddings)
    return docsearch


# Page config
st.set_page_config(page_title='youtube-gpt', page_icon='Img/robot.webp')

# Auth
st.sidebar.image("Img/robot.webp")
api_key_env = os.environ.get("OPENAI_API_KEY")

if api_key_env:
    st.sidebar.write("`Found OPENAI_API_KEY in environment.`")
else:
    api_key = st.sidebar.text_input("`OpenAI API Key:`", type="password")
    os.environ["OPENAI_API_KEY"] = api_key

st.sidebar.write("`GitHub:` [youtube-gpt](https://github.com/wrnu/youtube-gpt)")
st.sidebar.write("`Forked from:` [doc-gpt](https://github.com/PineappleExpress808/doc-gpt)")
st.sidebar.write("`Original By:` [@RLanceMartin](https://twitter.com/RLanceMartin)")
chunk_chars = st.sidebar.radio("`Choose chunk size for splitting`", (2000, 3000, 4000), index=1)
st.sidebar.info("`Larger chunk size can produce better answers, but may high ChatGPT context limit (4096 tokens)`")

# App
st.header("youtube-gpt")
st.info("`Hello! I am ChatGPT, linked to the YouTube video URL that you provide below.`", icon="👋")

youtube_url = st.text_input("`YouTube URL:` ")
is_valid_url = re.match(r"^(https?://)?(m|www)?\.youtube\.com/watch\?v=.{11}", youtube_url)
if youtube_url and not is_valid_url:
    st.error("Please enter a valid YouTube URL")
if youtube_url and not (api_key_env or api_key):
    st.error("Please enter a valid OpenAI API key")

st.info("`Ask me a question about the video and I will try to answer it.`", icon="🤖")

if youtube_url and (api_key_env or api_key):
    # Split and create index
    d = split_youtube(youtube_url, chunk_chars, overlap=chunk_chars // 10)
    ix = create_ix(d)
    # Use ChatGPT with index QA chain
    llm = OpenAIChat(temperature=0)
    chain_type_kwargs = {"prompt": PROMPT}
    chain = VectorDBQA.from_chain_type(llm, chain_type="stuff", vectorstore=ix, chain_type_kwargs=chain_type_kwargs)
    query = st.text_input("`Please ask a question:` ", "Create a summary, a table of contents and a summary of each item")
    try:
        st.info("%s" % chain.run(query))
    except openai.error.InvalidRequestError:
        # Limitation w/ ChatGPT: 4096 token context length
        # https://github.com/acheong08/ChatGPT/discussions/649
        st.warning('Error with model request, often due to context length. Try reducing chunk size.', icon="⚠️")
