import logging
import os
from dotenv import load_dotenv
#woocomerce_extension main class 
from woocommerce import API
import sqlite3
import json 
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.llms import openai
from langchain_community.tools import Tool
from langchain.agents import initialize_agent, AgentType
import threading
from queue import Queue
import time
import requests
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI # Import ChatOpenAI from langchain_community
from langchain_core.tools import tool
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

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
    def __init__(self, httpsurl=None, wckey=None, wcsecret=None, db_path=None ):
        # Load from environment if not provided
        httpsurl = httpsurl or os.getenv("WC_URL")
        wckey = wckey or os.getenv("WC_KEY")
        wcsecret = wcsecret or os.getenv("WC_SECRET")
        db_path = db_path or os.getenv("DB_PATH", "woocomerce.db")
        super().__init__(
            url=httpsurl,
            consumer_key=wckey,
            consumer_secret=wcsecret,
            version="wc/v3",
            wp_api=True
        )
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_table()
        self.all_products = None  # for access from any where in code

    def _create_table(self):
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
           id INTEGER PRIMARY KEY,
           name TEXT,
           description TEXT,
           variations TEXT,
           price TEXT,
           regular_price TEXT,
           sale_price TEXT,
           on_sale boolean,
           total_sale Text,
           stock_quantity INTEGER,
           stock_status TEXT,
           category TEXT,
           image_url TEXT)""")
        
        self.conn.commit()

    def fetch_site_data(self):  # getting all products data from site (name, description, image, price, ...)
        
        self.all_products = []
        page = 1

        while True:
            response = self.get("products", params={"per_page": 100, "page": page})
            products = response.json()

            if not products:  # وقتی دیگه محصولی برنگشت
                break

            self.all_products.extend(products)
            page += 1

        #return self.all_products 
        logger.info("Fetching complete")
    
    def put_db_products(self): # import important data to database
      
      for p in self.all_products:
       product_id = p["id"]
       name = p["name"]
       description = p["description"]
       price = p["price"]
      # regular_price =p["regular_price"] if "regular_price" in p else ""
      # sale_price = p["sale_price"] if "sale_price" in p else ""
     #  on_sale = p["on_sale"]
      # total_sale = p["total_sale"] if "total_sale" in p else ""
       #stock_quantity= p[stock_quantity] if "stock_quantity" in p else ""
      # stock = p["stock_status"]
     #  category = p["categories"][0]["name"] if p["categories"] else "بدون دسته"
      # image_url = p["images"][0]["src"] if p["images"] else ""

      # self.cursor.execute("""
       #    INSERT OR REPLACE INTO products (id, name,description, price,regular_price, sale_price,on_sale,total_sale, stock_quantity, stock_status, category, image_url)
       #    VALUES (?, ?, ?, ?, ?, ?)
        #   """, (product_id, name, description ,price , regular_price, sale_price ,on_sale, total_sale, stock_quantity, stock, category, image_url))


       self.cursor.execute("""
           INSERT OR REPLACE INTO products (id, name,description, price)
           VALUES (?, ?, ?, ?)
           """, (product_id, name, description ,price ))


      self.conn.commit()
      logger.info("Data inserted to database")

    def generate_variation_text(self):
       
        # SET API KEY
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        MODEL_NAME = os.getenv("MODEL_NAME")
        # General settings
        MAX_WORKERS = 5  # Number of concurrent threads
        SLEEP_TIME = 1.5  # Sleep time between requests

          
        llm = ChatOpenAI(
         model_name=MODEL_NAME,
         openai_api_base="https://openrouter.ai/api/v1",
         openai_api_key=OPENROUTER_API_KEY,
        )

        # Prompt template for generating different versions of product descriptions
        prompt = PromptTemplate.from_template(
            """This is a product description:
                        "{description}"
                            Please write {n} different and persuasive versions of this text for an online store. The tone should be friendly and convincing."""
                        )

        # Output as plain text
        chain = prompt | llm | StrOutputParser()

        def generate_variations( description, n=2):
          try:
           output = chain.invoke({"description": description, "n": n})
           logger.info(f"Generated variations for description: {description[:30]}...")
           return output
          except Exception as e:
           logger.error(f"LangChain Error: {str(e)}")
           raise

        # FUNCTION TO EXECUTE PER PRODUCT IN THREAD
        def process_product(product_id, description):
           try:
             variations = generate_variations( description)
             
             self.cursor.execute(
              "INSERT INTO products (product_id, variation_text) VALUES (?, ?)",
               (product_id, variations)
              )
             self.conn.commit()
             self.conn.close()
             logger.info(f"Saved: Product {product_id}")
             time.sleep(SLEEP_TIME)

           except Exception as e:
             logger.error(f"Error for product {product_id}: {str(e)}")

        # Queue system for multithreading management
        def worker():
         while True:
           item = q.get()
           if item is None:
             break
           product_id, description = item
           process_product(product_id, description)
           q.task_done()

         # Read products from database
         self.cursor.execute("SELECT id, description FROM products")
         products = self.cursor.fetchall()
         self.conn.close()

        # Create threads and queue
         q = Queue()
         threads = []

         for i in range(MAX_WORKERS):
          t = threading.Thread(target=worker)
          t.start()
          threads.append(t)

        # Add tasks to queue
         for product in products:
          q.put(product)

        # Wait for all tasks to finish
         q.join()

        # Close threads
         for i in range(MAX_WORKERS):
           q.put(None)
           for t in threads:
             t.join()

         logger.info("All products processed.")


   


    def compute_embedding(self):


        # Initial settings
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        MODEL_NAME = os.getenv("EMBEDDING_MODEL")
        NUM_THREADS = 4

       
        # Prepare output table
        def setup_output_table():
            
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS text_embeddings (
                    id INTEGER,
                    description TEXT,
                    embedding TEXT
                )
            """)
            self.conn.commit()
            self.conn.close()

        setup_output_table()

        # define agent tool
        @tool
        def embed_and_store(id: int, description: str, variations: str) -> str:
            """Generate embedding for text and store it in the database."""
            try:
                embedder = OpenAIEmbeddings(
                    model=MODEL_NAME,
                    openai_api_key=OPENROUTER_API_KEY,
                    base_url="https://openrouter.ai/api/v1"
                )

                vector_des = embedder.embed_query(description)
                vector_var = embedder.embed_query(variations)
               
                self.cursor.execute("INSERT INTO text_embeddings (id, description, embedding) VALUES (?, ?, ?)",
                        (id, json.dumps(vector_des), json.dumps(vector_var)))
                self.conn.commit()
                self.conn.close()
                logger.info(f"Saved embedding: id={id}")
                return f"Saved: id={id} "
            except Exception as e:
                logger.error(f"Embedding error: {str(e)}")
                return f"Error: {str(e)}"

        # Thread and Queue system
        q = Queue()

        def worker(agent):
            while True:
                item = q.get()
                if item is None:
                    break
                row_id, des, var = item
                agent.invoke({"input": f"Save embedding for text1 and text2:\n{des}\n{var}", "id": row_id, "description": "des", "variations": var})
               
                q.task_done()
                time.sleep(1)

        # Read data
        
        self.cursor.execute("SELECT id, text1, text2 FROM text_data")
        rows = self.cursor.fetchall()
        self.conn.close()

        # Build the agent
        tools = [embed_and_store]
        agent = initialize_agent(
            tools=tools,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

        # Start threads
        threads = []
        for _ in range(NUM_THREADS):
            t = threading.Thread(target=worker, args=(agent,))
            t.start()
            threads.append(t)

        # Fill the queue
        for row in rows:
            q.put(row)

        q.join()

        # Stop threads
        for _ in range(NUM_THREADS):
            q.put(None)
        for t in threads:
            t.join()

        logger.info("All embeddings processed and saved.")

   

# Example usage
if __name__ == "__main__":
    processor = woocomerce_ext()
    processor.fetch_site_data()
    processor.put_db_products()
    # processor.generate_variation_text()
      
