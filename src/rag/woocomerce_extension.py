import logging
import os
from dotenv import load_dotenv
import json
import time
from woocommerce import API
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.llms import openai
from langchain_community.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
import psycopg2

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Load environment variables from .env file
load_dotenv()

class woocomerce_ext(API):
    def __init__(self, httpsurl=None, wckey=None, wcsecret=None):
        httpsurl = httpsurl or os.getenv("WC_URL")
        wckey = wckey or os.getenv("WC_KEY")
        wcsecret = wcsecret or os.getenv("WC_SECRET")
        super().__init__(
            url=httpsurl,
            consumer_key=wckey,
            consumer_secret=wcsecret,
            version="wc/v3",
            wp_api=True
        )
        self._create_table()
        self.all_products = None

    def _get_conn(self):
        return psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )

    def _create_table(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS products (
                   id INTEGER PRIMARY KEY,
                   name TEXT,
                   description TEXT,
                   variations TEXT,
                   price TEXT,
                   regular_price TEXT,
                   sale_price TEXT,
                   on_sale BOOLEAN,
                   total_sale TEXT,
                   stock_quantity INTEGER,
                   stock_status TEXT,
                   category TEXT,
                   image_url TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS text_embeddings (
                    id INTEGER,
                    description TEXT,
                    embedding TEXT
                )
            """)
            conn.commit()
            cursor.close()

    def fetch_site_data(self):
        self.all_products = []
        page = 1
        while True:
            response = self.get("products", params={"per_page": 100, "page": page})
            products = response.json()
            if not products:
                break
            self.all_products.extend(products)
            page += 1
        logger.info("Fetching complete")

    def put_db_products(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            for p in self.all_products:
                product_id = p["id"]
                name = p["name"]
                description = p["description"]
                price = p["price"]
                cursor.execute("""
                    INSERT INTO products (id, name, description, price)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name, description=EXCLUDED.description, price=EXCLUDED.price
                """, (product_id, name, description, price))
            conn.commit()
            cursor.close()
        logger.info("Data inserted to database")

    def generate_variation_text(self):
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        MODEL_NAME = os.getenv("MODEL_NAME")
        SLEEP_TIME = 1.5
        llm = ChatOpenAI(
            model_name=MODEL_NAME,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=OPENROUTER_API_KEY,
        )
        prompt = PromptTemplate.from_template(
            """This is a product description:
                        "{description}"
                            Please write {n} different and persuasive versions of this text for an online store. The tone should be friendly and convincing."""
        )
        chain = prompt | llm | StrOutputParser()
        def generate_variations(description, n=2):
            try:
                output = chain.invoke({"description": description, "n": n})
                logger.info(f"Generated variations for description: {description[:30]}...")
                return output
            except Exception as e:
                logger.error(f"LangChain Error: {str(e)}")
                raise
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, description FROM products")
            products = cursor.fetchall()
            cursor.close()
        for product_id, description in products:
            try:
                variations = generate_variations(description)
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE products SET variations = %s WHERE id = %s",
                        (variations, product_id)
                    )
                    conn.commit()
                    cursor.close()
                logger.info(f"Saved: Product {product_id}")
                time.sleep(SLEEP_TIME)
            except Exception as e:
                logger.error(f"Error for product {product_id}: {str(e)}")
        logger.info("All products processed.")

    def compute_embedding(self):
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        MODEL_NAME = os.getenv("EMBEDDING_MODEL")
        @tool
        def embed_and_store(id: int, description: str, variations: str) -> str:
            try:
                embedder = OpenAIEmbeddings(
                    model=MODEL_NAME,
                    openai_api_key=OPENROUTER_API_KEY,
                    base_url="https://openrouter.ai/api/v1"
                )
                vector_des = embedder.embed_query(description)
                vector_var = embedder.embed_query(variations)
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO text_embeddings (id, description, embedding) VALUES (%s, %s, %s)",
                            (id, json.dumps(vector_des), json.dumps(vector_var)))
                    conn.commit()
                    cursor.close()
                logger.info(f"Saved embedding: id={id}")
                return f"Saved: id={id} "
            except Exception as e:
                logger.error(f"Embedding error: {str(e)}")
                return f"Error: {str(e)}"
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, description, variations FROM products")
            rows = cursor.fetchall()
            cursor.close()
        tools = [embed_and_store]
        agent = initialize_agent(
            tools=tools,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        for row in rows:
            row_id, des, var = row
            agent.invoke({"input": f"Save embedding for text1 and text2:\n{des}\n{var}", "id": row_id, "description": des, "variations": var})
            time.sleep(1)
        logger.info("All embeddings processed and saved.")

# Example usage
if __name__ == "__main__":
    processor = woocomerce_ext()
    processor.fetch_site_data()
    processor.put_db_products()
    # processor.generate_variation_text()
      
