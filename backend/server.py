from fastapi import FastAPI, APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import hashlib
import jwt
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import PyPDF2
import io
from sentence_transformers import SentenceTransformer
import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Configure Gemini
genai.configure(api_key="AIzaSyA3dRlGjSFwwKjCnq1vgaHfrMx36mJE22c")
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Create the main app
app = FastAPI()
api_router = APIRouter(prefix="/api")
security = HTTPBearer()

# Models
class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class User(BaseModel):
    user_id: str
    username: str
    api_key: str
    created_at: datetime

class Document(BaseModel):
    id: str
    user_id: str
    filename: str
    content: str
    chunks: List[str]
    embeddings: List[List[float]]
    upload_time: datetime
    chunk_count: int
    status: str

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]

# Utility functions - SIMPLIFIED
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_token(user_id: str) -> str:
    # Super simple token - just prefix + user_id
    return f"simple_token_{user_id}"

def verify_token(token: str) -> str:
    # Super simple token verification - just check if it starts with our prefix
    if token and token.startswith("simple_token_"):
        return token.replace("simple_token_", "")
    raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    return verify_token(credentials.credentials)

def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1
        
        if current_size >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def get_embeddings(texts: List[str]) -> List[List[float]]:
    embeddings = embedding_model.encode(texts)
    return embeddings.tolist()

def find_relevant_chunks(query: str, document_chunks: List[str], document_embeddings: List[List[float]], top_k: int = 3) -> List[dict]:
    query_embedding = embedding_model.encode([query])
    similarities = cosine_similarity(query_embedding, document_embeddings)[0]
    
    # Get top k most similar chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.3:  # Minimum similarity threshold
            results.append({
                'chunk_index': int(idx),
                'content': document_chunks[idx],
                'relevance_score': float(similarities[idx])
            })
    
    return results

# SIMPLIFIED Authentication endpoints
@api_router.post("/auth/register")
async def register(user_data: UserCreate):
    # Check if user exists
    existing_user = await db.users.find_one({"username": user_data.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Create new user - SIMPLE STORAGE
    user_id = str(uuid.uuid4())
    api_key = f"sk-docubrain-{uuid.uuid4().hex[:20]}"
    
    user = {
        "user_id": user_id,
        "username": user_data.username,
        "password": user_data.password,  # Store plain password for simplicity
        "api_key": api_key,
        "created_at": datetime.utcnow()
    }
    
    await db.users.insert_one(user)
    token = create_token(user_id)
    
    return {
        "success": True,
        "message": "Registration successful!",
        "user_id": user_id,
        "token": token,
        "api_key": api_key
    }

@api_router.post("/auth/login")
async def login(user_data: UserLogin):
    # Find user
    user = await db.users.find_one({"username": user_data.username})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Simple password check
    if user_data.password != user["password"]:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    token = create_token(user["user_id"])
    
    return {
        "success": True,
        "message": "Login successful!",
        "user_id": user["user_id"],
        "token": token,
        "api_key": user["api_key"]
    }

# Document endpoints
@api_router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user)
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Read file content
    content = await file.read()
    text = extract_text_from_pdf(content)
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")
    
    # Process document
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)
    
    # Save to database
    doc_id = str(uuid.uuid4())
    document = {
        "id": doc_id,
        "user_id": user_id,
        "filename": file.filename,
        "content": text,
        "chunks": chunks,
        "embeddings": embeddings,
        "upload_time": datetime.utcnow(),
        "chunk_count": len(chunks),
        "status": "completed"
    }
    
    await db.documents.insert_one(document)
    
    return {"message": "Document uploaded and processed successfully", "document_id": doc_id}

@api_router.post("/documents/text")
async def add_text_document(
    title: str = Form(...),
    content: str = Form(...),
    user_id: str = Depends(get_current_user)
):
    if not content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")
    
    # Process text
    chunks = chunk_text(content)
    embeddings = get_embeddings(chunks)
    
    # Save to database
    doc_id = str(uuid.uuid4())
    document = {
        "id": doc_id,
        "user_id": user_id,
        "filename": f"{title}.txt",
        "content": content,
        "chunks": chunks,
        "embeddings": embeddings,
        "upload_time": datetime.utcnow(),
        "chunk_count": len(chunks),
        "status": "completed"
    }
    
    await db.documents.insert_one(document)
    
    return {"message": "Text document processed successfully", "document_id": doc_id}

@api_router.get("/documents")
async def get_documents(user_id: str = Depends(get_current_user)):
    documents = await db.documents.find(
        {"user_id": user_id}
    ).to_list(100)
    
    # Remove heavy fields for response
    for doc in documents:
        doc.pop("content", None)
        doc.pop("chunks", None) 
        doc.pop("embeddings", None)
        doc.pop("_id", None)
    
    return documents

# Query endpoint
@api_router.post("/query", response_model=QueryResponse)
async def query_documents(query: QueryRequest, user_id: str = Depends(get_current_user)):
    # Get user documents
    documents = await db.documents.find({"user_id": user_id}).to_list(100)
    
    if not documents:
        raise HTTPException(status_code=400, detail="No documents found. Please upload some documents first.")
    
    # Find relevant chunks across all documents
    all_relevant_chunks = []
    
    for doc in documents:
        relevant_chunks = find_relevant_chunks(
            query.question, 
            doc["chunks"], 
            doc["embeddings"]
        )
        
        for chunk in relevant_chunks:
            chunk['filename'] = doc['filename']
            all_relevant_chunks.append(chunk)
    
    if not all_relevant_chunks:
        return QueryResponse(
            answer="I couldn't find relevant information in your documents to answer this question.",
            sources=[]
        )
    
    # Sort by relevance and take top 5
    all_relevant_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
    top_chunks = all_relevant_chunks[:5]
    
    # Create context for Gemini
    context = "\n\n".join([chunk['content'] for chunk in top_chunks])
    
    prompt = f"""Based on the following context from the user's documents, answer the question. Only use information from the provided context.

Context:
{context}

Question: {query.question}

Answer:"""
    
    try:
        response = gemini_model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        answer = f"Error generating response: {str(e)}"
    
    # Prepare sources
    sources = [
        {
            "filename": chunk["filename"],
            "chunk_index": chunk["chunk_index"],
            "relevance_score": chunk["relevance_score"]
        }
        for chunk in top_chunks
    ]
    
    return QueryResponse(answer=answer, sources=sources)

# External API endpoint
@api_router.post("/external/query")
async def external_query(
    api_key: str = Form(...),
    question: str = Form(...)
):
    # Find user by API key
    user = await db.users.find_one({"api_key": api_key})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Use the regular query logic
    query_request = QueryRequest(question=question)
    return await query_documents(query_request, user["user_id"])

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()