# main.py

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag_system import RAGSystem
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Gazette RAG API")

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    main_sources: list[dict]
    other_sources: list[dict]
# Initialize RAG once at startup
rag_system = RAGSystem()

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    result = rag_system.query(request.query)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@app.get("/")
async def root():
    return {"message": "Visit /static/index.html for the frontend"}