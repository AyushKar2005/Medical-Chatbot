from flask import Flask, render_template, request
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from src.prompt import *
import os

app = Flask(__name__)
load_dotenv()


PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY


llm = None
retriever = None
rag_chain = None

def init_chain():
    global llm, retriever, rag_chain

    if not llm:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.4,
            max_output_tokens=500,
            google_api_key=GEMINI_API_KEY
        )

    if not retriever:
        from src.helper import download_hugging_face_embeddings
        embeddings = download_hugging_face_embeddings()

        docsearch = PineconeVectorStore.from_existing_index(
            index_name="medicalbot",
            embedding=embeddings,
        )
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    if not rag_chain:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    init_chain()  
    msg = request.form["msg"]
    response = rag_chain.invoke({"input": msg})
    return str(response.get("answer", "Sorry, I couldn't understand."))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
