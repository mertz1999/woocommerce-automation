import os
from openai import AzureOpenAI
from db_init import get_conn
from logger import logger

def search_products_by_text(query_text, top_k=10):
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
        dimensions=DIMENSIONS
    )
    embedding = response.data[0].embedding
    emb_str = "[" + ", ".join(map(str, embedding)) + "]"
    conn = get_conn()
    cursor = conn.cursor()
    sql = """
        SELECT p.name, p.price, p.id as product_id, pd.id as variation_id,
               (pe.embedding <=> %s::vector) AS distance
        FROM product_embeddings pe
        JOIN product_descriptions pd ON pe.variation_id = pd.id
        JOIN products p ON pd.product_id = p.id
        ORDER BY distance ASC
        LIMIT 30;
    """
    cursor.execute(sql, (emb_str,))
    results = cursor.fetchall()
    conn.close()
    best_by_product = {}
    for row in results:
        name, price, product_id, variation_id, distance = row
        if product_id not in best_by_product or distance < best_by_product[product_id][4]:
            best_by_product[product_id] = row
    final_results = sorted(best_by_product.values(), key=lambda x: x[4])[:top_k]
    logger.info(f"Search for '{query_text}' returned {len(final_results)} results.")
    return [
        {"name": r[0], "price": r[1], "product_id": r[2], "variation_id": r[3], "distance": r[4]}
        for r in final_results
    ] 