import streamlit as st
from transformers import pipeline, set_seed
import pyttsx3
import tempfile
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
import os

# Initialize TTS engine
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Load the text-generation pipeline once and cache it
@st.cache_resource
def load_generator():
    # Use a model compatible with text generation; DialoGPT supports text-generation
    generator = pipeline("text-generation", model="microsoft/DialoGPT-medium")
    set_seed(42)
    return generator

# Create vectorstore from text for document QA
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

# Format chat history to prompt string for the model
def build_prompt(history, new_user_input):
    prompt = ""
    for speaker, text in history:
        if speaker == "User":
            prompt += f"User: {text}\n"
        else:
            prompt += f"Assistant: {text}\n"
    prompt += f"User: {new_user_input}\nAssistant:"
    return prompt

# Run generation with max tokens and reasonable parameters
def generate_response(generator, prompt):
    outputs = generator(
        prompt,
        max_length=len(prompt.split()) + 60,  # context + new tokens, adjust as needed
        num_return_sequences=1,
        pad_token_id=50256,  # for GPT-2 based models
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        eos_token_id=50256
    )
    text = outputs[0]["generated_text"]
    # Extract only the assistant's response (remove input prompt)
    response = text[len(prompt):].strip()
    # Cut off any trailing incomplete sentences after first newline
    if "\n" in response:
        response = response.split("\n")[0].strip()
    return response

# Streamlit app
st.set_page_config(
    page_title="Updated Offline AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Updated Offline Personal AI Assistant")
st.markdown("Local chat + Document Q&A + TTS — No API, no internet needed!")

# Sidebar controls
tts_enable = st.sidebar.checkbox("Speak AI responses", value=True)
theme = st.sidebar.selectbox("Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""<style>body {background-color: #111; color: #eee !important;}</style>""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of tuples: (speaker, text)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_uploaded" not in st.session_state:
    st.session_state.doc_uploaded = False

generator = load_generator()

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

# User input area
user_input = st.text_input("Your message:", key="input")
ask_doc = st.session_state.doc_uploaded and st.checkbox("Ask about uploaded document", value=False)

if st.button("Send") and user_input.strip():
    user_text = user_input.strip()
    st.session_state.chat_history.append(("User", user_text))

    if ask_doc and st.session_state.vectorstore is not None:
        # Query document vectorstore
        response = get_doc_answer(st.session_state.vectorstore, user_text)
        response = "(Doc Q&A) " + response
    else:
        # Build prompt and generate response
        prompt = build_prompt(st.session_state.chat_history[:-1], user_text)
        response = generate_response(generator, prompt)

    st.session_state.chat_history.append(("Assistant", response))

    if tts_enable:
        speak_text(response)

# Display chat history
with st.expander("💬 Conversation History", expanded=True):
    for speaker, text in st.session_state.chat_history:
        if speaker == "User":
            st.markdown(f"**🧑 You:** {text}")
        else:
            st.markdown(f"**🤖 Assistant:** {text}")

st.info("🔒 100% local — runs fully offline on your machine!")
