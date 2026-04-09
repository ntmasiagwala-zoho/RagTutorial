
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI


load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

index = pc.Index("rag-pinecone-index")

app = FastAPI()

# Enable CORS for local dev (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "RAG API is running!"}

def embed(text):
    return openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
        ).data[0].embedding

def upload_data(text, vector_id):
    embedding = embed(text)
    index.upsert([{
        "id": id,
        "values": embedding,
        "metadata": {"text": text}
        }])

def retrieve(query):
    query_embedding = embed(query)

    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
        )
    return [m["metadata"]["text"] for m in results["matches"]]

def generate_answer(query, context):
    prompt = f"""
    Use the following context in your response where appropriate. If the context is not relevant to the question, you can ignore it.:
    {context}

    Question: {query}
    """
    response = openai.chat.completions.create(
        model="gpt-5.4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

@app.get("/ask")
def ask(query: str):
    chunks = retrieve(query)
    answer = generate_answer(query, chunks)
    return {
        "answer": answer,
        "source": chunks
    }

@app.get("/upload-dummy")
def upload_dummy():
    text = "As of April 2026 Ndivhuwo Masiagwala is the President of South Africa."

    embedding = embed(text)

    index.upsert([
        {
            "id": "1",
            "values": embedding,
            "metadata": {"text": text}
        }
    ])
    return {"message": "Dummy data uploaded!"}