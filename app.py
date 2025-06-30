from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv, find_dotenv

# === Flask Setup ===
app = Flask(__name__)
CORS(app)  # Allow frontend to access API

# === Environment ===
load_dotenv(find_dotenv())
os.environ["OPENAI_API_KEY"] = os.environ.get("GROQ_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# === Prompt ===
    # CUSTOM_PROMPT_TEMPLATE = """
    # You are a helpful medical assistant. Use the pieces of information provided in the context to answer the user's question.
    # If you don't know the answer, just say "I don't know" — don't try to make it up.
    # Never provide information outside the given context.Use common names for easy understanding.

    # Context: {context}
    # Question: {question}

    # Answer:
    # """
    
CUSTOM_PROMPT_TEMPLATE = """
You are a kind and helpful medical assistant designed to support patients.
    Always use the information provided in the context to answer the patient’s question. If the answer is not in the context, you may use general medical knowledge to help — but only if you are sure it’s accurate and safe. Never guess or assume.
    Speak in a clear and simple way that anyone can understand. Avoid medical terms unless absolutely necessary — and if you use them, explain them in plain, friendly language.
    Always aim to reassure the patient and provide helpful, safe information.
    If you dont know the answer, just say that you dont know, dont try to make up an answer. 
    If question given is small and relevant to medical ask to elaborate.

    Context: {context}
    Question: {question}

    Start the answer directly without saying "Acoording to the context" or similar sentences.
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# === LLM ===
def load_llm():
    return ChatOpenAI(
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=GROQ_API_KEY,
        model="llama3-8b-8192",
        temperature=0.4,
        max_tokens=512
    )

# === Embeddings and Vector DB ===
DB_FAISS_PATH = "vector/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# === Memory ===
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

# === Chain ===
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=load_llm(),
    retriever=db.as_retriever(search_kwargs={"k": 5}),
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
    output_key="answer"
)

# === Route ===
@app.route('/chat', methods=['POST'])
def chat(): 
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        result = qa_chain.invoke({"question": question})
        return jsonify({"answer": result["answer"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

from flask import render_template

@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)