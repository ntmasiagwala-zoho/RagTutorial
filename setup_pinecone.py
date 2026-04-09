from pinecone import Pinecone
from pinecone import ServerlessSpec

pc = Pinecone(api_key="pcsk_54TzCh_2YQEXFVJnUWsqZok7G4Z88mackfTkfH3RAaUVtahGwJkpXMkAj8Jrwpehiw7edQ")

pc.create_index(
    name="rag-pinecone-index",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )