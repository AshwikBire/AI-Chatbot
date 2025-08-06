import streamlit as st
from transformers import pipeline, Conversation
import pyttsx3
import tempfile
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
import os

#### --- Utility Functions --- ####

# Initialize (fast) TTS engine
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Load chatbot model
@st.cache_resource
def load_chatbot():
    return pipeline("conversational", model="microsoft/DialoGPT-medium")

# Embed and store uploaded docs for QA
@st.cache_resource
def create_vectorstore_from_text(text):
    splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
    docs = [d.page_content for d in splitter.create_documents([text])]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_texts(docs, embeddings)
    return vs

def get_doc_answer(vectorstore, user_query):
    docs_and_scores = vectorstore.similarity_search_with_score(user_query, k=2)
    if docs_and_scores:
        return docs_and_scores[0][0].page_content.strip()
    else:
        return "Sorry, I couldn't find relevant information in your uploaded document."

#### --- UI Setup --- ####

# Page config
st.set_page_config(
    page_title="Supercharged Offline AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Supercharged Personal AI Assistant")
st.markdown(
    """
    <small>
    <b>Features:</b> Conversational AI, PDF/text file Q&A, text-to-speech, no APIs. <br>
    <b>How to use:</b> Ask questions, or upload your docs and ask about them!
    </small>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("Settings")
tts_enable = st.sidebar.checkbox("Speak AI responses", value=True)
theme = st.sidebar.selectbox("Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""<style>body {background-color: #111; color: #eee !important;}</style>""", unsafe_allow_html=True)

#### --- State --- ####

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_uploaded" not in st.session_state:
    st.session_state.doc_uploaded = False

#### --- Model and Chat --- ####

chatbot = load_chatbot()

#### --- Document Upload/Q&A --- ####

uploaded_file = st.file_uploader("📄 Upload a PDF or text file for question answering (optional)", type=["pdf", "txt"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name.split('.')[-1]) as tf:
        tf.write(uploaded_file.read())
        temp_filepath = tf.name
    if uploaded_file.name.endswith('.pdf'):
        loader = PyPDFLoader(temp_filepath)
    else:
        loader = TextLoader(temp_filepath)
    docs = loader.load()
    text = "\n".join([d.page_content for d in docs])
    st.session_state.vectorstore = create_vectorstore_from_text(text)
    st.session_state.doc_uploaded = True
    st.success(f"Document '{uploaded_file.name}' processed for Q&A!")
    os.remove(temp_filepath)

#### --- Chat Interface --- ####

user_input = st.text_input("Type your message:", key="input")
ask_file = st.session_state.doc_uploaded and st.checkbox("Ask about uploaded doc", value=False)

if st.button("Send") and user_input.strip():
    st.session_state.chat_history.append(("You", user_input))

    if ask_file and st.session_state.vectorstore:
        answer = get_doc_answer(st.session_state.vectorstore, user_input)
        response = f"(Doc Q&A) {answer}"
    else:
        conversation = Conversation(user_input)
        output = chatbot(conversation)
        response = output.generated_responses[-1]
    st.session_state.chat_history.append(("Assistant", response))

    if tts_enable:
        speak_text(response)

# Chat history display
with st.expander("💬 Conversation History", expanded=True):
    for speaker, msg in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            st.markdown(f"**🤖 Assistant:** {msg}")

st.info("🔒 Runs locally. No data leaves your machine. Perfect for personal and confidential tasks!")

