from supabase import create_client, Client
from sentence_transformers import SentenceTransformer, util
import re
import os
from openai import AzureOpenAI
import json
from logger import logger
import db_init
from converting import extract_price_and_duration,fa_to_en



def searchByFilter(query_text, top_k=10):

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
 response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query_text],
        dimensions=DIMENSIONS,
        timeout= 60
        )

 embedding = response.data[0].embedding
 emb_query = "[" + ", ".join(map(str, embedding)) + "]"


# connect to database
 conn = db_init.get_conn()
 cursor = conn.cursor()


#gain price and duration from user query and converting to standard formt
 
 
 price, duration = extract_price_and_duration(query_text)
 

 logger.info(f"price or durtion mentioned in query: '{price}' and '{duration}'")
 

 # Build the SQL query with proper parameterization
 if price is not None and duration is not None:
     find_products = """
        SELECT id
        FROM products
        WHERE (meta_data ->> 'price')::numeric <= %s
         AND meta_data ->> 'course_time' ILIKE %s;
     """
     cursor.execute(find_products, (price, duration))
     filtered_ids = cursor.fetchall()
 elif price is not None:
     find_products = """
        SELECT id
        FROM products
        WHERE (meta_data ->> 'price')::numeric <= %s;
     """
     cursor.execute(find_products, (price,))
     filtered_ids = cursor.fetchall()
 elif duration is not None:
    find_products = """
    SELECT id
    FROM products
    WHERE meta_data ->> 'course_time' ILIKE %s;
"""
    cursor.execute(find_products, (duration,))
    filtered_ids = cursor.fetchall()
 else:
    # No filters, get all products
    find_products = "SELECT id FROM products;"
    cursor.execute(find_products)
    filtered_ids = cursor.fetchall()
 print(filtered_ids)
 logger.info(f" filtered product id: {filtered_ids}")
 
 find_embedding = """SELECT p.name,
       p.price,
       p.id AS product_id,
       (pe.embedding <=> %s) AS distance
 FROM product_embeddings pe
 JOIN product_descriptions pd ON pd.id = pe.variation_id
 JOIN products p ON p.id = pd.product_id
 WHERE p.id = ANY(%s::int[])
 ORDER BY distance ASC
 LIMIT 30;"""

 cursor.execute(find_embedding, (emb_query, filtered_ids))
 fetched_embedding = cursor.fetchall()
 conn.close()

 best_by_product = {}
 for row in fetched_embedding:
        name, price, product_id, distance = row
        if product_id not in best_by_product or distance < best_by_product[product_id][4]:
            best_by_product[product_id] = row
 final_results = sorted(best_by_product.values(), key=lambda x: x[4])[:top_k]
 logger.info(f"Search for '{query_text}' returned {len(final_results)} results.")
 return [
        {"name": r[0], "price": r[1], "product_id": r[2], "variation_id": r[3], "distance": r[4]}
        for r in final_results
    ] 




