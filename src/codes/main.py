import json
from db_init import create_tables
from woocommerce_data import fetch_site_data, put_db_products
from embedding import compute_embedding
from search import search_products_by_text
from woocommerce import API
from dotenv import load_dotenv
import os

def main():
    # Load environment variables
    load_dotenv()

    # Initialize DB
    create_tables()

    # Setup WooCommerce API (credentials from env)
    # api = API(
    #     url=os.getenv("WC_URL"),
    #     consumer_key=os.getenv("WC_KEY"),
    #     consumer_secret=os.getenv("WC_SECRET"),
    #     version="wc/v3",
    #     wp_api=True
    # )

    # Fetch and insert products
    # products = fetch_site_data(api)
    # put_db_products(json_path='product.json')

    
    # Compute embeddings
    compute_embedding(batch_size=5)

    # Search
    results = search_products_by_text("فیلم برداری حرفه ای")
    for r in results:
        print(r)
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main() 