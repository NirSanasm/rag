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
import logging
import re

# --- Logging Setup ---
# Set the log level. Use DEBUG for verbose output, INFO for standard operation.
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Environment & API Key Loading ---
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

# --- Database Configuration ---

def get_db_config():
    """Parses the Neon DB URL from environment variables."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL environment variable is not set")
        raise ValueError("DATABASE_URL environment variable is not set")
    
    parsed = urlparse(db_url)
    logger.debug("Database config parsed successfully.")
    return {
        "dbname": parsed.path[1:],
        "user": parsed.username,
        "password": parsed.password,
        "host": parsed.hostname,
        "port": parsed.port or 5432,
        "sslmode": "require"  # Neon requires SSL
    }

# --- Data Structures ---

@dataclass
class ProcessedQuery:
    """Dataclass to hold the structured output from the LLM query processor."""
    bm25_keywords: list[str]
    intent: str
    gazetteno: Optional[str]
    date_filter: Optional[dict]
    vector_search_query: str

    def to_dict(self):
        return asdict(self)

# --- Core Classes ---

class LLMQueryProcessor:
    """Processes the raw user query into a structured ProcessedQuery object using an LLM."""
    
    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        # Use Dependency Injection for the OpenAI client
        self.client = client
        self.model = model
        self.current_year = datetime.now().year
        logger.debug(f"LLMQueryProcessor initialized with model: {self.model}")

    def process_query(self, user_query: str) -> ProcessedQuery:
        """
        Sends the user query to the LLM for structuring, parsing, and expansion.
        """
        logger.info(f"Processing user query: '{user_query}'")
        prompt = self._build_prompt(user_query)
        logger.debug(f"LLM prompt for query processing:\n{prompt}")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a query processing assistant for a gazette search system. You must return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result_json = response.choices[0].message.content
            logger.debug(f"LLM response JSON: {result_json}")
            
            result = json.loads(result_json)
            
            processed_query = ProcessedQuery(
                bm25_keywords=result.get("bm25_keywords", []),
                intent=result.get("intent", "general"),
                gazetteno=result.get("gazetteno"),
                date_filter=result.get("date_filter"),
                vector_search_query=result.get("vector_search_query", user_query)
            )
            logger.info(f"Query processed successfully: {processed_query.to_dict()}")
            return processed_query

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode LLM response JSON: {e}\nResponse was: {result_json}")
            # Fallback to a general query
            return ProcessedQuery(bm25_keywords=[], intent="general", gazetteno=None, date_filter=None, vector_search_query=user_query)
        except Exception as e:
            logger.error(f"Error during LLM query processing: {e}", exc_info=True)
            # Fallback to a general query
            return ProcessedQuery(bm25_keywords=[], intent="general", gazetteno=None, date_filter=None, vector_search_query=user_query)

    def _build_prompt(self, user_query: str) -> str:
        """Helper method to construct the LLM prompt."""
        return f"""Extract structured information from this gazette search query.
Current year: {self.current_year}

Query: "{user_query}"

Return JSON with:
1. bm25_keywords: Core keywords without stopwords. Normalized spelling must add 1-2 relevant synonyms for main conceptual keywords. Combine all into a flat list.
2. intent: One of [fact_finding, summarization, monitoring, comparison, general]
3. gazetteno: Gazette number if present, else null
4. date_filter: If date is specified, Parse dates as {{"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}} or if not specified, null
5. vector_search_query: Original query + 2-3 semantic variations combined as a single string.

Return ONLY valid JSON."""


class RAGSystem:
    """The main RAG system, orchestrating retrieval, ranking, and generation."""
    
    # --- Configuration Constants ---
    BM25_LIMIT = 150
    VECTOR_LIMIT = 20
    RERANK_TOP_N = 6

    def __init__(self):
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY is not set.")
            raise ValueError("OPENAI_API_KEY is not set.")
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.query_processor = LLMQueryProcessor(client=self.client)
        self.db_config = get_db_config() # Get config once
        logger.info("RAGSystem initialized successfully.")

    def _get_connection(self):
        """Returns a new database connection (must be used in a 'with' block)"""
        try:
            conn = psycopg2.connect(**self.db_config)
            logger.debug("Database connection established.")
            return conn
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to database: {e}", exc_info=True)
            raise

    def get_nomic_embedding(self, texts: List[str], task_type: str = "search_query") -> List[List[float]]:
        """Generates embeddings for a list of texts using the Nomic API."""
        if not NOMIC_API_KEY:
            logger.error("NOMIC_API_KEY not configured.")
            raise ValueError("NOMIC_API_KEY not configured")
        
        logger.debug(f"Generating Nomic embeddings for {len(texts)} text(s)...")
        try:
            response = requests.post(
                "https://api-atlas.nomic.ai/v1/embedding/text",
                headers={
                    "Authorization": f"Bearer {NOMIC_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"texts": texts, "task_type": task_type, "dimensionality": 768},
                timeout=10 # Add a timeout
            )
            response.raise_for_status()  # Raises HTTPError for bad responses
            logger.debug("Nomic embeddings generated successfully.")
            return response.json()["embeddings"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Nomic API request failed: {e}", exc_info=True)
            raise

    def bm25_search_with_filters(self, processed_query: ProcessedQuery, limit: int = 50) -> List[str]:
        """Performs a BM25 (full-text) search with optional filters for date and gazette number."""
        if not processed_query.bm25_keywords:
            logger.warning("No BM25 keywords provided. Skipping BM25 search.")
            return []

        # Use ' | ' (OR) for keywords, as generated by the LLM prompt
        tsquery = " | ".join(processed_query.bm25_keywords)
        logger.info(f"Performing BM25 search with query: '{tsquery}'")
        
        # Base query parts
        # **BUG FIX**: Changed plainto_tsquery to to_tsquery to respect the '|' OR operator
        base_sql = """
            SELECT gazette_id
            FROM gazettes
            WHERE to_tsvector('english', ocr_text) @@ to_tsquery('english', %s)
        """
        params = [tsquery]

        # Add gazette number filter if present
        if processed_query.gazetteno is not None:
            base_sql += " AND gazetteno = %s"
            params.append(str(processed_query.gazetteno))
            logger.debug(f"Added filter: gazetteno = {processed_query.gazetteno} (as string)")

        # Add date range filter if present
        if processed_query.date_filter is not None:
            start_date = processed_query.date_filter.get("start_date")
            end_date = processed_query.date_filter.get("end_date")
            if start_date:
                base_sql += " AND publicationdate >= %s"
                params.append(start_date)
            if end_date:
                base_sql += " AND publicationdate <= %s"
                params.append(end_date)
            logger.debug(f"Added filter: date_filter = {processed_query.date_filter}")

        # Complete query with ordering and limit
        base_sql += """
            ORDER BY ts_rank_cd(to_tsvector('english', ocr_text), to_tsquery('english', %s)) DESC
            LIMIT %s
        """
        params.extend([tsquery, limit])

        logger.debug(f"BM25 SQL: {base_sql} \nParams: {params}")

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(base_sql, params)
                    results = [row[0] for row in cur.fetchall()]
                    logger.info(f"BM25 search found {len(results)} gazette IDs.")
                    return results
        except psycopg2.Error as e:
            logger.error(f"BM25 DB query failed: {e}", exc_info=True)
            return [] # Return empty list on failure

    def retrieve_relevant_chunks(self, query_embedding: List[float], gazette_ids: List[str] = None, limit: int = 10) -> List[Tuple[str, str, int, float]]:
        """Retrieves relevant text chunks using vector similarity search, optionally filtered by gazette_ids."""
        logger.info(f"Retrieving vector chunks. Filtering by {len(gazette_ids) if gazette_ids else '0'} gazette IDs.")
        embedding_str = json.dumps(query_embedding)
        
        # --- Refactored Query Building (DRY) ---
        base_sql = """
            SELECT content, gazette_id, chunk_index, (embedding <=> %s::vector) as distance
            FROM gazette_embeddings
        """
        params: list[any] = [embedding_str]
        
        where_clauses = []
        if gazette_ids:
            # Use IN operator for filtering
            placeholders = ','.join(['%s'] * len(gazette_ids))
            where_clauses.append(f"gazette_id IN ({placeholders})")
            params.extend(gazette_ids)
            
        if where_clauses:
            base_sql += " WHERE " + " AND ".join(where_clauses)
            
        # Add ordering and limit
        base_sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([embedding_str, limit])
        
        logger.debug(f"Vector search SQL: {base_sql}")
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(base_sql, params)
                    results = cur.fetchall()
                    logger.info(f"Vector search retrieved {len(results)} initial chunks.")
                    return results
        except psycopg2.Error as e:
            logger.error(f"Vector search DB query failed: {e}", exc_info=True)
            return []

    def rerank_chunks(self, query: str, documents: List[Tuple[str, str, int, float]], top_n: int = 3) -> List[Tuple[str, str, int]]:
        """Reranks the retrieved chunks using the Jina API for better relevance."""
        if not JINA_API_KEY:
            logger.error("JINA_API_KEY not configured.")
            raise ValueError("JINA_API_KEY not configured")
        
        if not documents:
            logger.warning("No documents provided to rerank.")
            return []

        logger.info(f"Reranking {len(documents)} chunks for query: '{query}'")
        doc_contents = [doc[0] for doc in documents]
        payload = {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "top_n": top_n,
            "documents": doc_contents
        }

        try:
            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {JINA_API_KEY}"},
                json=payload,
                timeout=10 # Add a timeout
            )
            response.raise_for_status()
            
            results = response.json().get("results", [])
            reranked = []
            for r in results:
                idx = r["index"]
                # (content, gazette_id, chunk_index)
                reranked.append((documents[idx][0], documents[idx][1], documents[idx][2]))
            
            logger.info(f"Reranked to {len(reranked)} chunks.")
            return reranked
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Jina API rerank failed: {e}", exc_info=True)
            # Fallback: return the top_n chunks from the original list (based on vector distance)
            return [(doc[0], doc[1], doc[2]) for doc in documents[:top_n]]
        
    #  added for sanitisation of context before sending to llm ( to prevent finish reason of "content_filter")
    def sanitize_context(self, text: str) -> str:
        text = re.sub(r"\b\d{5,}\b", "[number]", text)  # replace long digit sequences
        text = re.sub(r"http\S+|www\S+", "[url]", text)
        text = re.sub(r"[^ -~\n]", "", text)  # remove non-ASCII garbage
        return text




    def generate_answer(
        self,
        query: str,
        processed_query: ProcessedQuery,
        context_chunks: List[str],
        chunk_metadata: List[Tuple[str, int]],
    ) -> str:
        """Generates the final answer using an LLM, customized by user intent."""
        logger.info("Generating final answer...")

        if not context_chunks:
            logger.warning("No context provided for answer generation.")
            return "No relevant information found to answer the question."

        # Build the formatted context
        context_parts = []
        for chunk, (gazette_id, chunk_idx) in zip(context_chunks, chunk_metadata):
            context_parts.append(f"Source [Gazette: {gazette_id}, Chunk: {chunk_idx}]:\n{chunk}")
        context = "\n\n".join(context_parts)

        context = self.sanitize_context(context)

        

        # --- Intent-based instructions ---
        intent = processed_query.intent.lower() if processed_query.intent else "general"

        if intent == "fact_finding":
            user_prompt = (
                f"You are tasked with providing a **factual and precise** answer based only on the context. "
                f"Do not speculate. Clearly cite the source for each fact using the format "
                f"[Gazette: gazette_id, Chunk: chunk_index].\n\n"
                f"Context:\n{context}\n\nQuestion: {query}"
            )

        elif intent == "summarization":
            user_prompt = (
                f"Summarize the key points relevant to the question using the provided context. "
                f"Focus on clarity and completeness. Avoid repetition. "
                f"Include brief source citations in the format [Gazette: gazette_id, Chunk: chunk_index].\n\n"
                f"Context:\n{context}\n\nQuestion: {query}"
            )

        elif intent == "monitoring":
            user_prompt = (
                f"Analyze the provided context to identify **updates, changes, or progress over time** "
                f"relevant to the question. Highlight trends or recurring themes. "
                f"Support each point with a source citation [Gazette: gazette_id, Chunk: chunk_index].\n\n"
                f"Context:\n{context}\n\nQuestion: {query}"
            )

        elif intent == "comparison":
            user_prompt = (
                f"Compare and contrast the relevant items, events, or rules mentioned in the context. "
                f"Clearly list similarities and differences. "
                f"Provide supporting citations for each point using [Gazette: gazette_id, Chunk: chunk_index].\n\n"
                f"Context:\n{context}\n\nQuestion: {query}"
            )

        else:  # general
            user_prompt = (
                f"Answer the question concisely using the provided context. "
                f"Make sure the answer is helpful and grounded in the context. "
                f"Cite the sources using [Gazette: gazette_id, Chunk: chunk_index].\n\n"
                f"Context:\n{context}\n\nQuestion: {query}"
            )

        # --- Call LLM ---
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a domain expert assistant that answers strictly based on the provided gazette context.",
                    },
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )

            answer = response.choices[0].message.content.strip()
            logger.info("Answer generated successfully.")
            logger.info(response)

            logger.debug(f"Answer preview: {answer[:300]}...")
            return answer

        except Exception as e:
            logger.error(f"OpenAI answer generation failed: {e}", exc_info=True)
            return "An error occurred while generating the answer."


    def _fetch_gazette_metadata(self, gazette_ids: List[str]) -> dict:
        """Fetches metadata for a list of gazette_ids for enriching the source references."""
        if not gazette_ids:
            logger.warning("No gazette IDs provided to fetch metadata for.")
            return {}
        
        logger.debug(f"Fetching metadata for {len(gazette_ids)} unique gazette IDs.")
        placeholders = ','.join(['%s'] * len(gazette_ids))
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT gazette_id, title, gazetteno, notificationdate, gazettedoc
                        FROM gazettes
                        WHERE gazette_id IN ({placeholders})
                    """, gazette_ids)
                    
                    rows = cur.fetchall()
                    logger.debug(f"Fetched metadata for {len(rows)} gazettes.")
                    return {
                        row[0]: {
                            "title": row[1],
                            "gazetteno": row[2],
                            "notificationdate": row[3],
                            "gazettedoc": row[4]
                        }
                        for row in rows
                    }
        except psycopg2.Error as e:
            logger.error(f"Metadata fetch DB query failed: {e}", exc_info=True)
            return {}

    def _enrich_sources(self, sources_raw: List[dict], gazette_metadata: dict) -> List[dict]:
        """Helper function to combine raw source data with fetched metadata."""
        enriched_sources = []
        for src in sources_raw:
            meta = gazette_metadata.get(src["gazette_id"], {})
            enriched_sources.append({
                **src,
                "title": meta.get("title"),
                "gazetteno": meta.get("gazetteno"),
                "notificationdate": meta.get("notificationdate"),
                "gazettedoc": meta.get("gazettedoc")
            })
        return enriched_sources

    def query(self, user_query: str) -> dict:
        """
        Main entry point for the RAG system.
        Orchestrates the entire process from query -> retrieve -> rerank -> generate.
        """
        logger.info(f"--- RAG Query Started --- \nQuery: '{user_query}'")
        try:
            # 1. Process Query
            processed_query = self.query_processor.process_query(user_query)
            
            # 2. Retrieve (Hybrid Search)
            #    Stage 1: Keyword (BM25) pre-filtering
            gazette_ids = self.bm25_search_with_filters(processed_query, limit=self.BM25_LIMIT)
            
            #    Stage 2: Vector Search (on filtered IDs or all chunks)
            query_embedding = self.get_nomic_embedding([processed_query.vector_search_query])[0]
            
            if gazette_ids:
                # Hybrid: Vector search within BM25-filtered gazettes
                initial_chunks = self.retrieve_relevant_chunks(query_embedding, gazette_ids, limit=self.VECTOR_LIMIT)
            else:
                # Fallback: Pure vector search
                logger.warning("No BM25 results. Falling back to pure vector search.")
                initial_chunks = self.retrieve_relevant_chunks(query_embedding, limit=self.VECTOR_LIMIT)

            if not initial_chunks:
                logger.warning("No relevant information found after retrieval step.")
                return {
                    "answer": "No relevant information found.",
                    "main_sources": [],
                    "other_sources": []
                }

            # 3. Rerank
            reranked = self.rerank_chunks(user_query, initial_chunks, top_n=self.RERANK_TOP_N)
            if not reranked:
                # Handle reranker failure (e.g., API down)
                logger.warning("Reranking failed or returned empty. Using top vector search results.")
                # Use top N from initial_chunks as fallback
                reranked = [(doc[0], doc[1], doc[2]) for doc in initial_chunks[:self.RERANK_TOP_N]]

            # 4. Generate
            context_chunks = [c[0] for c in reranked]
            metadata = [(c[1], c[2]) for c in reranked]
            answer = self.generate_answer(user_query, processed_query, context_chunks, metadata)

            # 5. Collate Sources
            main_sources_raw = [
                {"gazette_id": gazette_id, "chunk_index": chunk_index, "context": content}
                for content, gazette_id, chunk_index in reranked
            ]

            main_keys = {(gazette_id, chunk_index) for _, gazette_id, chunk_index in reranked}
            other_sources_raw = [
                {"gazette_id": gazette_id, "chunk_index": chunk_index, "context": content}
                for content, gazette_id, chunk_index, _ in initial_chunks
                if (gazette_id, chunk_index) not in main_keys
            ]

            # 6. Fetch Metadata and Enrich
            all_gazette_ids = list({src["gazette_id"] for src in main_sources_raw + other_sources_raw})
            gazette_metadata = self._fetch_gazette_metadata(all_gazette_ids)

            main_sources = self._enrich_sources(main_sources_raw, gazette_metadata)
            other_sources = self._enrich_sources(other_sources_raw, gazette_metadata)

            logger.info("--- RAG Query Completed Successfully ---")
            return {
                "answer": answer,
                "main_sources": main_sources,
                "other_sources": other_sources
            }

        except Exception as e:
            # Catch-all for any unhandled exceptions in the pipeline
            logger.error(f"Unhandled exception in RAG query: {e}", exc_info=True)
            return {
                "answer": "An critical error occurred while processing your query.",
                "main_sources": [],
                "other_sources": []
            }

# # --- Example Usage (for testing) ---
# if __name__ == "__main__":
#     logger.info("Starting RAG system for direct execution...")
    
#     # Check for necessary API keys
#     if not all([OPENAI_API_KEY, NOMIC_API_KEY, JINA_API_KEY, os.getenv("DATABASE_URL")]):
#         logger.error("Missing one or more required environment variables: OPENAI_API_KEY, NOMIC_API_KEY, JINA_API_KEY, DATABASE_URL")
#     else:
#         try:
#             rag_system = RAGSystem()
            
#             test_query = "find chief secretary in the gazette for the year 2012?"
#             # test_query = "Gazette number 52 regarding employee promotions"
            
#             print(f"\nQuerying system with: '{test_query}'")
            
#             result = rag_system.query(test_query)
            
#             print("\n--- RESULT ---")
#             print(f"Answer:\n{result['answer']}")
            
#             print(f"\nMain Sources ({len(result['main_sources'])}):")
#             for i, src in enumerate(result['main_sources']):
#                 print(f"  {i+1}. Gazette: {src.get('gazetteno', src['gazette_id'])}, Chunk: {src['chunk_index']}")
#                 print(f"     Title: {src.get('title', 'N/A')}")
#                 print(f"     Context: {src['context'][:100]}...")

#             print(f"\nOther Sources ({len(result['other_sources'])}):")
#             if result['other_sources']:
#                 src = result['other_sources'][0]
#                 print(f"  1. Gazette: {src.get('gazetteno', src['gazette_id'])}, Chunk: {src['chunk_index']}")
#                 print(f"     Title: {src.get('title', 'N/A')}")
            
#         except Exception as e:
#             logger.error(f"Failed to run RAG system example: {e}", exc_info=True)