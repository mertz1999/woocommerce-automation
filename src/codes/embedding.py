import os
import time
from tqdm import tqdm
from openai import AzureOpenAI
from db_init import get_conn
from psycopg2.extras import execute_values
from logger import logger

def compute_embedding(batch_size):
    API_KEY = os.getenv("API_KEY")
    API_VERSION = os.getenv("API_VERSION")
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
    EMBEDDING_MODEL = "text-embedding-3-large"
    DIMENSIONS = 256
    client = AzureOpenAI(
        api_key=API_KEY,
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT
    )
    def fetch_batch_from_db(batch_size):
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, variation FROM product_descriptions
            WHERE id NOT IN (SELECT variation_id FROM product_embeddings)
            LIMIT %s
        """, (batch_size,))
        rows = cursor.fetchall()
        conn.close()
        return rows
    def save_embedding_to_db(results):
        conn = get_conn()
        cursor = conn.cursor()
        valid_results = [(var_id, embedding) for var_id, embedding in results if embedding is not None]
        if valid_results:
            execute_values(
                cursor,
                """
                INSERT INTO product_embeddings (variation_id, embedding)
                VALUES %s
                ON CONFLICT (variation_id) DO NOTHING
                """,
                valid_results
            )
            conn.commit()
            logger.info("A batch of embeddings saved to db")
        cursor.close()
        conn.close()
    def embed_batch(records):
        texts = []
        var_ids = []
        MAX_CHARS = 10000
        for id, text in records:
            if len(text) > MAX_CHARS:
                logger.warning(f"Variation id {id} text too long ({len(text)} chars), truncating to {MAX_CHARS} chars.")
                text = text[:MAX_CHARS]
            texts.append(text)
            var_ids.append(id)
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
            dimensions=DIMENSIONS
        )
        embeddings = [item.embedding for item in response.data]
        return list(zip(var_ids, embeddings))
    batch_num = 1
    while True:
        records = fetch_batch_from_db(batch_size)
        if not records:
            logger.info("All records have been embedded.")
            break
        logger.info(f"\n Processing batch {batch_num} with {len(records)} records...")
        results = embed_batch(records)
        save_embedding_to_db(results)
        logger.info(f"✅ Batch {batch_num} saved.")
        batch_num += 1
        time.sleep(1) 