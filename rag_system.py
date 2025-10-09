# rag_system.py

import os
import psycopg2
import requests
from openai import OpenAI
from typing import List, Tuple, Optional
from dotenv import load_dotenv
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from urllib.parse import urlparse

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


def get_db_config():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    parsed = urlparse(db_url)
    return {
        "dbname": parsed.path[1:],
        "user": parsed.username,
        "password": parsed.password,
        "host": parsed.hostname,
        "port": parsed.port or 5432,
        "sslmode": "require"  # Neon requires SSL
    }


@dataclass
class ProcessedQuery:
    bm25_keywords: list[str]
    intent: str
    department: Optional[str]
    gazetteno: Optional[str]
    date_filter: Optional[dict]
    vector_search_query: str

    def to_dict(self):
        return asdict(self)


class LLMQueryProcessor:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.current_year = datetime.now().year

    def process_query(self, user_query: str) -> ProcessedQuery:
        prompt = self._build_prompt(user_query)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a query processing assistant for a gazette search system. You must return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return ProcessedQuery(
            bm25_keywords=result.get("bm25_keywords", []),
            intent=result.get("intent", "general"),
            department=result.get("department"),
            gazetteno=result.get("gazetteno"),
            date_filter=result.get("date_filter"),
            vector_search_query=result.get("vector_search_query", user_query)
        )

    def _build_prompt(self, user_query: str) -> str:
        return f"""Extract structured information from this gazette search query.
Current year: {self.current_year}

Query: "{user_query}"

Return JSON with:
1. bm25_keywords: Core keywords without stopwords, with synonyms, normalized spelling
2. intent: One of [fact_finding, summarization, monitoring, comparison, general]
3. department: Department name if mentioned, else null
4. gazetteno: Gazette number if present, else null
5. date_filter: Parse dates as {{"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}} or null
6. vector_search_query: Original + 3 semantic variations combined

Return ONLY valid JSON."""


class RAGSystem:
    def __init__(self):
        self.query_processor = LLMQueryProcessor(api_key=OPENAI_API_KEY)

    def _get_connection(self):
        """Returns a new database connection (must be used in a 'with' block)"""
        config = get_db_config()
        return psycopg2.connect(**config)

    def get_nomic_embedding(self, texts: List[str], task_type: str = "search_query") -> List[List[float]]:
        if not NOMIC_API_KEY:
            raise ValueError("NOMIC_API_KEY not configured")
        response = requests.post(
            "https://api-atlas.nomic.ai/v1/embedding/text",
            headers={
                "Authorization": f"Bearer {NOMIC_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"texts": texts, "task_type": task_type, "dimensionality": 768}
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    def bm25_search_with_filters(self, processed_query: ProcessedQuery, limit: int = 50) -> List[str]:
        if not processed_query.bm25_keywords:
            return []
        tsquery = " | ".join(processed_query.bm25_keywords)
        sql = """
            SELECT gazette_id
            FROM gazettes
            WHERE to_tsvector('english', ocr_text) @@ plainto_tsquery(%s)
            ORDER BY ts_rank_cd(to_tsvector('english', ocr_text), plainto_tsquery(%s)) DESC
            LIMIT %s
        """
        params = [tsquery, tsquery, limit]
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return [row[0] for row in cur.fetchall()]

    def retrieve_relevant_chunks(self, query_embedding: List[float], gazette_ids: List[str] = None, limit: int = 10) -> List[Tuple[str, str, int, float]]:
        embedding_str = json.dumps(query_embedding)
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if gazette_ids:
                    placeholders = ','.join(['%s'] * len(gazette_ids))
                    cur.execute(f"""
                        SELECT content, gazette_id, chunk_index, (embedding <=> %s::vector) as distance
                        FROM gazette_embeddings 
                        WHERE gazette_id IN ({placeholders})
                        ORDER BY embedding <=> %s::vector 
                        LIMIT %s
                    """, (embedding_str, *gazette_ids, embedding_str, limit))
                else:
                    cur.execute("""
                        SELECT content, gazette_id, chunk_index, (embedding <=> %s::vector) as distance
                        FROM gazette_embeddings 
                        ORDER BY embedding <=> %s::vector 
                        LIMIT %s
                    """, (embedding_str, embedding_str, limit))
                return cur.fetchall()

    def rerank_chunks(self, query: str, documents: List[Tuple[str, str, int, float]], top_n: int = 3) -> List[Tuple[str, str, int]]:
        if not JINA_API_KEY:
            raise ValueError("JINA_API_KEY not configured")
        if not documents:
            return []
        doc_contents = [doc[0] for doc in documents]
        payload = {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "top_n": top_n,
            "documents": doc_contents
        }
        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {JINA_API_KEY}"},
            json=payload
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        reranked = []
        for r in results:
            idx = r["index"]
            reranked.append((documents[idx][0], documents[idx][1], documents[idx][2]))
        return reranked

    def generate_answer(self, query: str, context_chunks: List[str], chunk_metadata: List[Tuple[str, int]]) -> str:
        context_parts = []
        for chunk, (gazette_id, chunk_idx) in zip(context_chunks, chunk_metadata):
            context_parts.append(f"Source [Gazette: {gazette_id}, Chunk: {chunk_idx}]:\n{chunk}")
        context = "\n\n".join(context_parts)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer based only on the provided context. Cite sources."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()

    def query(self, user_query: str) -> dict:
        try:
            processed_query = self.query_processor.process_query(user_query)
            gazette_ids = self.bm25_search_with_filters(processed_query, limit=50)
            query_embedding = self.get_nomic_embedding([processed_query.vector_search_query])[0]
            
            if gazette_ids:
                initial_chunks = self.retrieve_relevant_chunks(query_embedding, gazette_ids, limit=10)
            else:
                initial_chunks = self.retrieve_relevant_chunks(query_embedding, limit=10)

            if not initial_chunks:
                return {"answer": "No relevant information found.", "sources": []}

            reranked = self.rerank_chunks(user_query, initial_chunks, top_n=3)
            context_chunks = [c[0] for c in reranked]
            metadata = [(c[1], c[2]) for c in reranked]

            answer = self.generate_answer(user_query, context_chunks, metadata)
            sources = [{"gazette_id": gid, "chunk_index": idx} for gid, idx in metadata]

            return {"answer": answer, "sources": sources}

        except Exception as e:
            return {"error": str(e)}