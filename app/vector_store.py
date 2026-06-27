from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import GOOGLE_API_KEY

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=GOOGLE_API_KEY
)

def get_vectorstore(chunks):

    db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="vectorstore"
    )

    return db