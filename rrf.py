import os
import psycopg2
import requests
from openai import OpenAI
from typing import List, Tuple, Optional, Dict, Any
from dotenv import load_dotenv
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from urllib.parse import urlparse
import logging
from collections import defaultdict

# --- Logging Setup ---
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
            return ProcessedQuery(bm25_keywords=[], intent="general", gazetteno=None, date_filter=None, vector_search_query=user_query)
        except Exception as e:
            logger.error(f"Error during LLM query processing: {e}", exc_info=True)
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
    """
    The main RAG system.
    Orchestrates retrieval (parallel keyword + vector search), 
    fusion (RRF), ranking (Jina), and generation (OpenAI).
    """
    
    # --- Configuration Constants ---
    # Number of results to fetch from each retriever
    INITIAL_SEARCH_LIMIT = 50 
    # Number of fused results to send to the reranker
    RERANK_INPUT_LIMIT = 20 
    # Final number of chunks for context
    RERANK_TOP_N = 6 
    # RRF constant
    RRF_K = 60 

    def __init__(self):
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY is not set.")
            raise ValueError("OPENAI_API_KEY is not set.")
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.query_processor = LLMQueryProcessor(client=self.client)
        self.db_config = get_db_config()
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
                timeout=10
            )
            response.raise_for_status()
            logger.debug("Nomic embeddings generated successfully.")
            return response.json()["embeddings"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Nomic API request failed: {e}", exc_info=True)
            raise

    def _build_filter_sql(self, processed_query: ProcessedQuery) -> Tuple[str, List[Any]]:
        """
        Helper to build the JOIN and WHERE clauses for filtering by gazetteno and date.
        This is used by both bm25_chunk_search and vector_chunk_search.
        """
        params = []
        # We must join with gazettes table to filter on its metadata
        sql = " FROM gazette_embeddings ge JOIN gazettes g ON ge.gazette_id = g.gazette_id"
        where_clauses = []

        if processed_query.gazetteno is not None:
            where_clauses.append("g.gazetteno = %s")
            params.append(processed_query.gazetteno)
            logger.debug(f"Added filter: gazetteno = {processed_query.gazetteno}")

        if processed_query.date_filter is not None:
            start_date = processed_query.date_filter.get("start_date")
            end_date = processed_query.date_filter.get("end_date")
            if start_date:
                where_clauses.append("g.publicationdate >= %s")
                params.append(start_date)
            if end_date:
                where_clauses.append("g.publicationdate <= %s")
                params.append(end_date)
            logger.debug(f"Added filter: date_filter = {processed_query.date_filter}")
        
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
            
        return sql, params

    def bm25_chunk_search(self, processed_query: ProcessedQuery, limit: int) -> List[Tuple[str, str, int]]:
        """
        Performs a BM25 (full-text) search on the CHUNK CONTENT.
        Returns a ranked list of (content, gazette_id, chunk_index).
        """
        if not processed_query.bm25_keywords:
            logger.warning("No BM25 keywords provided. Skipping BM25 search.")
            return []

        tsquery = " | ".join(processed_query.bm25_keywords)
        logger.info(f"Performing BM25 chunk search with query: '{tsquery}'")

        filter_sql, filter_params = self._build_filter_sql(processed_query)
        
        # Add the FTS WHERE clause
        fts_sql = "to_tsvector('english', ge.content) @@ to_tsquery('english', %s)"
        if not filter_params:
            filter_sql += " WHERE " + fts_sql
        else:
            filter_sql += " AND " + fts_sql
        
        params = filter_params + [tsquery]
        
        base_sql = "SELECT ge.content, ge.gazette_id, ge.chunk_index"
        
        # Complete query with ordering and limit
        full_sql = base_sql + filter_sql + """
            ORDER BY ts_rank_cd(to_tsvector('english', ge.content), to_tsquery('english', %s)) DESC
            LIMIT %s
        """
        params.extend([tsquery, limit])

        logger.debug(f"BM25 Chunk SQL: {full_sql} \nParams: {params}")

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(full_sql, params)
                    results = cur.fetchall()
                    logger.info(f"BM25 chunk search found {len(results)} chunks.")
                    return results
        except psycopg2.Error as e:
            logger.error(f"BM25 chunk DB query failed: {e}", exc_info=True)
            return []

    def vector_chunk_search(self, query_embedding: List[float], processed_query: ProcessedQuery, limit: int) -> List[Tuple[str, str, int]]:
        """
        Performs a vector search on chunks, with filtering.
        Returns a ranked list of (content, gazette_id, chunk_index).
        """
        logger.info("Performing vector chunk search...")
        embedding_str = json.dumps(query_embedding)

        filter_sql, filter_params = self._build_filter_sql(processed_query)
        params = [embedding_str] + filter_params
        
        base_sql = "SELECT ge.content, ge.gazette_id, ge.chunk_index, (ge.embedding <=> %s::vector) as distance"
        
        full_sql = base_sql + filter_sql + """
            ORDER BY distance ASC
            LIMIT %s
        """
        params.append(limit)

        logger.debug(f"Vector Chunk SQL: {full_sql}")

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(full_sql, params)
                    # We only return the first 3 columns (content, id, index)
                    results = [(row[0], row[1], row[2]) for row in cur.fetchall()]
                    logger.info(f"Vector chunk search found {len(results)} chunks.")
                    return results
        except psycopg2.Error as e:
            logger.error(f"Vector chunk DB query failed: {e}", exc_info=True)
            return []

    def rrf_fusion(self, ranked_lists: List[List[Tuple[str, int]]], k: int) -> List[Tuple[str, int]]:
        """
        Combines multiple ranked lists using Reciprocal Rank Fusion (RRF).
        Expects lists of hashable items (e.g., (gazette_id, chunk_index) tuples).
        """
        logger.info(f"Performing RRF fusion on {len(ranked_lists)} lists.")
        scores = defaultdict(float)
        
        for ranked_list in ranked_lists:
            for rank, item in enumerate(ranked_list):
                # RRF formula: score += 1 / (k + rank)
                # We use rank + 1 because rank is 0-indexed
                scores[item] += 1 / (k + rank + 1)
        
        # Sort by the new RRF score in descending order
        sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        
        logger.debug(f"RRF fused {sum(len(l) for l in ranked_lists)} items into {len(sorted_items)}.")
        return [item for item, score in sorted_items]

    def rerank_chunks(self, query: str, documents: List[Tuple[str, str, int]], top_n: int) -> List[Tuple[str, str, int]]:
        """Ranks chunks using Jina API. Input is (content, gazette_id, chunk_index)."""
        if not JINA_API_KEY:
            logger.error("JINA_API_KEY not configured.")
            raise ValueError("JINA_API_KEY not configured")
        
        if not documents:
            logger.warning("No documents provided to rerank.")
            return []

        logger.info(f"Reranking {len(documents)} chunks for query: '{query}'")
        # Extract just the content for the API
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
                timeout=10
            )
            response.raise_for_status()
            
            results = response.json().get("results", [])
            reranked = []
            for r in results:
                original_index = r["index"]
                # Re-associate the original metadata
                reranked.append(documents[original_index])
            
            logger.info(f"Reranked to {len(reranked)} chunks.")
            return reranked
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Jina API rerank failed: {e}", exc_info=True)
            # Fallback: return the top_n chunks from the original fused list
            return documents[:top_n]

    def generate_answer(self, query: str, context_chunks: List[str], chunk_metadata: List[Tuple[str, int]]) -> str:
        """Generates the final answer using an LLM, based on the reranked context."""
        logger.info("Generating final answer...")
        if not context_chunks:
            logger.warning("No context provided for answer generation.")
            return "No relevant information found to answer the question."

        context_parts = []
        for chunk, (gazette_id, chunk_idx) in zip(context_chunks, chunk_metadata):
            context_parts.append(f"Source [Gazette: {gazette_id}, Chunk: {chunk_idx}]:\n{chunk}")
        
        context = "\n\n".join(context_parts)
        logger.debug(f"Context for answer generation:\n{context[:500]}...")

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer the user's question based *only* on the provided context. Clearly cite the source for each piece of information using the format [Gazette: gazette_id, Chunk: chunk_index]."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ],
                temperature=0.2
            )
            answer = response.choices[0].message.content.strip()
            logger.info("Answer generated successfully.")
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
        Orchestrates: 1. Process -> 2. Retrieve (Parallel) -> 3. Fuse (RRF) -> 4. Rerank -> 5. Generate
        """
        logger.info(f"--- RAG Query Started --- \nQuery: '{user_query}'")
        try:
            # 1. Process Query
            processed_query = self.query_processor.process_query(user_query)
            query_embedding = self.get_nomic_embedding([processed_query.vector_search_query])[0]

            # 2. Retrieve (Parallel Chunk-Level Search)
            bm25_results = self.bm25_chunk_search(processed_query, limit=self.INITIAL_SEARCH_LIMIT)
            vector_results = self.vector_chunk_search(query_embedding, processed_query, limit=self.INITIAL_SEARCH_LIMIT)

            # Create a map to store content by (gazette_id, chunk_index)
            # This avoids duplicating content and makes RRF keys simple
            chunk_content_map: Dict[Tuple[str, int], str] = {}
            for content, gazette_id, chunk_index in bm25_results + vector_results:
                key = (gazette_id, chunk_index)
                if key not in chunk_content_map:
                    chunk_content_map[key] = content
            
            # Create ranked lists of keys for RRF
            bm25_ranked_keys = [(r[1], r[2]) for r in bm25_results]
            vector_ranked_keys = [(r[1], r[2]) for r in vector_results]

            # 3. Fuse (RRF)
            fused_ranked_keys = self.rrf_fusion([bm25_ranked_keys, vector_ranked_keys], k=self.RRF_K)

            if not fused_ranked_keys:
                logger.warning("No relevant information found after RRF fusion.")
                return {"answer": "No relevant information found.", "main_sources": [], "other_sources": []}
            
            # Reconstruct the fused list with content, limited to RERANK_INPUT_LIMIT
            initial_chunks_for_rerank = []
            for gazette_id, chunk_index in fused_ranked_keys[:self.RERANK_INPUT_LIMIT]:
                content = chunk_content_map.get((gazette_id, chunk_index))
                if content:
                    initial_chunks_for_rerank.append((content, gazette_id, chunk_index))
            
            logger.info(f"Sending {len(initial_chunks_for_rerank)} fused chunks to reranker.")

            # 4. Rerank
            reranked = self.rerank_chunks(user_query, initial_chunks_for_rerank, top_n=self.RERANK_TOP_N)

            # 5. Generate
            context_chunks = [c[0] for c in reranked]
            metadata = [(c[1], c[2]) for c in reranked]
            answer = self.generate_answer(user_query, context_chunks, metadata)

            # 6. Collate Sources
            main_sources_raw = [
                {"gazette_id": gazette_id, "chunk_index": chunk_index, "context": content}
                for content, gazette_id, chunk_index in reranked
            ]

            main_keys = {(gazette_id, chunk_index) for _, gazette_id, chunk_index in reranked}
            # "Other" sources are from the fused list but *not* in the final reranked list
            other_sources_raw = [
                {"gazette_id": gazette_id, "chunk_index": chunk_index, "context": content}
                for content, gazette_id, chunk_index in initial_chunks_for_rerank
                if (gazette_id, chunk_index) not in main_keys
            ]

            # 7. Fetch Metadata and Enrich
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
            logger.error(f"Unhandled exception in RAG query: {e}", exc_info=True)
            return {
                "answer": "An critical error occurred while processing your query.",
                "main_sources": [],
                "other_sources": []
            }

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    logger.info("Starting RAG system for direct execution...")
    
    if not all([OPENAI_API_KEY, NOMIC_API_KEY, JINA_API_KEY, os.getenv("DATABASE_URL")]):
        logger.error("Missing one or more required environment variables: OPENAI_API_KEY, NOMIC_API_KEY, JINA_API_KEY, DATABASE_URL")
    else:
        try:
            rag_system = RAGSystem()
            
            test_query = "What are the new rules for land acquisition?"
            
            print(f"\nQuerying system with: '{test_query}'")
            
            result = rag_system.query(test_query)
            
            print("\n--- RESULT ---")
            print(f"Answer:\n{result['answer']}")
            
            print(f"\nMain Sources ({len(result['main_sources'])}):")
            for i, src in enumerate(result['main_sources']):
                print(f"  {i+1}. Gazette: {src.get('gazetteno', src['gazette_id'])}, Chunk: {src['chunk_index']}")
                print(f"     Title: {src.get('title', 'N/A')}")
                print(f"     Context: {src['context'][:100]}...")

            print(f"\nOther Sources ({len(result['other_sources'])}):")
            if result['other_sources']:
                src = result['other_sources'][0]
                print(f"  1. Gazette: {src.get('gazetteno', src['gazette_id'])}, Chunk: {src['chunk_index']}")
                print(f"     Title: {src.get('title', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Failed to run RAG system example: {e}", exc_info=True)