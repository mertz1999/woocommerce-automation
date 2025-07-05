import os
import json
import re
import html
from tqdm import tqdm
from woocommerce import API
from db_init import get_conn
from logger import logger

def fetch_site_data(api: API):
    all_products = []
    page = 1
    while True:
        response = api.get("products", params={"per_page": 100, "page": page})
        products = response.json()
        if not products:
            break
        all_products.extend(products)
        page += 1
    with open("product.json", "w", encoding="utf-8") as f:
        json.dump(all_products, f, ensure_ascii=False, indent=2)
    logger.info("Fetching complete")
    return all_products

def put_db_products(products=None, json_path=None):
    def clean_text(text):
        text = re.sub(r'<[^>]+>', '', text)
        text = html.unescape(text)
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    if json_path is not None:
        with open(json_path, 'r', encoding='utf-8') as f:
            products = json.load(f)

    with get_conn() as conn:
        cursor = conn.cursor()
        for p in tqdm(products, desc="Inserting products"):
            product_id = p["id"]
            name = clean_text(p["name"])
            price = p["price"]
            meta_data = json.dumps(p, ensure_ascii=False)
            cursor.execute(
                """
                INSERT INTO products (id, name, price, meta_data)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name, price=EXCLUDED.price, meta_data=EXCLUDED.meta_data
                """,
                (product_id, name, price, meta_data)
            )
            if "description" in p:
                description = clean_text(p["description"])
                cursor.execute(
                    """
                    INSERT INTO product_descriptions (product_id, variation)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (product_id, description)
                )
        conn.commit()
        cursor.close()
    logger.info("Data inserted to database") 