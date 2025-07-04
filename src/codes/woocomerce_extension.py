import logging
import os
from dotenv import load_dotenv
import numpy as np
import requests
import time
import re 
import html
from woocommerce import API
from langchain_community.llms import openai
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.output_parsers.structured import StructuredOutputParser
from langchain.output_parsers import ResponseSchema
from langchain_core.pydantic_v1 import BaseModel, Field
import psycopg2
from psycopg2.extras import execute_values
import concurrent.futures
import yaml
import json
from tqdm import tqdm
from openai import AzureOpenAI


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
        httpsurl =  httpsurl or os.getenv("WC_URL")
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
            host=os.getenv("DB_HOST")   ,     
            port= os.getenv("DB_PORT")  ,   
            database= os.getenv("DB_NAME") ,   
            user=os.getenv("DB_USER")     ,   
            password= os.getenv("DB_PASSWORD")  
        )
    #create database and tables that needed
    def _create_table(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            # Create new products table with updated structure
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS products (
                   id INTEGER PRIMARY KEY,
                   name TEXT,
                   price TEXT,
                   meta_data JSONB,
                   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Restore creation of other tables as before
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS product_descriptions (
                    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    product_id INT NOT NULL REFERENCES products(id),
                    variation TEXT,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS product_embeddings (
                    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    variation_id INT NOT NULL REFERENCES product_descriptions(id),
                    embedding vector(256),
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            cursor.close()
    # getting product information from site
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
        # Save products to product.json
        with open("product.json", "w", encoding="utf-8") as f:
            json.dump(self.all_products, f, ensure_ascii=False, indent=2)
    # inserting product information from site to database
    def put_db_products(self, json_path=None):
        def clean_text(text):
            text = re.sub(r'<[^>]+>', '', text)
            text = html.unescape(text)
            text = text.replace('\n', ' ').replace('\r', ' ')
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        # Load products from JSON if path is provided
        if json_path is not None:
            with open(json_path, 'r', encoding='utf-8') as f:
                products = json.load(f)
        else:
            products = self.all_products

        with self._get_conn() as conn:
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
                # Insert into product_descriptions if description exists
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
    # creating variations for product description
    def generate_variation_text(self):
        # Load Azure OpenAI credentials from .env
        API_KEY = os.getenv("API_KEY")
        API_VERSION = os.getenv("API_VERSION")
        AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
        MODEL_NAME = "gpt-4o-mini"
        SLEEP_TIME = 5
        MAX_WORKERS = 3  # Adjusted for your LLM rate limits
        with open("config.yaml", "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)
            num_of_var = config["inputParameter"]["num_of_variation"]

        client = AzureOpenAI(
            api_key=API_KEY,
            api_version=API_VERSION,
            azure_endpoint=AZURE_ENDPOINT
        )
        class VariationsSchema(BaseModel):
            variations: list[str] = Field(..., description="A list of short, complete product description variations.")

        # Define the response schema for variations
        variations_schema = ResponseSchema(name="variations", description="A list of short, complete product description variations.")
        output_parser = StructuredOutputParser(response_schemas=[variations_schema])

        # Update the prompt to instruct the model to return a JSON list of variations
        prompt_template = '''این توضیحات محصول است:
                        "{description}"
                        لطفا {n} نسخه متفاوت و کوتاه اما کامل از این متن را برای فروشگاه آنلاین بنویسید. هر نسخه باید مختصر باشد اما تمام اطلاعات مهم محصول را در بر بگیرد. لحن باید دوستانه و متقاعدکننده باشد. خروجی را فقط به صورت یک شیء JSON با کلید "variations" که مقدار آن یک لیست از متن‌ها است برگردان.
                        مثال خروجی:
                        {{"variations": ["متن ۱", "متن ۲", ...]}}
                        '''

        def generate_variations_for_product(product):
            product_id, description = product
            try:
                prompt = prompt_template.format(description=description, n=num_of_var)
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0.7,
                )
                raw_output = response.choices[0].message.content
                parsed = output_parser.parse(raw_output)
                variations = parsed["variations"]
                time.sleep(SLEEP_TIME)
                return (product_id, variations)
            except Exception as e:
                logger.error(f"Error for product {product_id}: {str(e)}")
                return (product_id, [])

        # Fetch all products first
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, meta_data FROM products")
            products = cursor.fetchall()
            cursor.close()

        # Prepare (id, description) tuples from meta_data
        product_tuples = []
        for prod_id, meta_data in products:
            try:
                meta = meta_data if isinstance(meta_data, dict) else json.loads(meta_data)
                description = meta.get('description', '')
                if description:
                    product_tuples.append((prod_id, description))
            except Exception as e:
                logger.error(f"Error parsing meta_data for product {prod_id}: {str(e)}")

        # Parallelize LLM calls
        updates = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_product = {executor.submit(generate_variations_for_product, product): product for product in product_tuples}
            for future in tqdm(concurrent.futures.as_completed(future_to_product), total=len(product_tuples), desc="Generating variations"):
                product_id, variations = future.result()
                for variation in variations:
                    if variation:
                        updates.append((product_id, variation))

        # Do all DB updates in one connection
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """INSERT INTO product_descriptions (product_id, variation)
                    VALUES (%s, %s)""",
                updates
            )
            conn.commit()
            cursor.close()
        logger.info("All products processed.")

    #computing embeding for description and variations
    def compute_embedding(self, batch_size):
        EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
        MODEL_NAME = os.getenv("EMBEDDING_MODEL")
        EMBBEDDING_URL = os.getenv("EMBEDDING_URL")

        def fetch_batch_from_db(batch_size):
           conn =self._get_conn()
           cursor = conn.cursor()
           cursor.execute("SELECT id, variation FROM product_descriptions LIMIT %s", (batch_size,))
           rows = cursor.fetchall()
           conn.close()
           return rows

        # store embeding in database
        def save_embedding_to_db(results):
            conn = self._get_conn()
            cursor = conn.cursor()
            # Filter out None embeddings
            valid_results = [(var_id, embedding) for var_id, embedding in results if embedding is not None]
            if valid_results:
                execute_values(
                    cursor,
                    """
                    INSERT INTO product_embeddings (variation_id, embedding)
                    VALUES %s
                    """,
                    valid_results
                )
                conn.commit()
                logger.info("A batch of embeddings saved to db")
            cursor.close()
            conn.close()

        #embeding process for one batch
        def embed_batch(records):
            texts= []
            var_ids=[]
            for id, text in records:
                texts.append(text)
                var_ids.append(id)
            headers = {
                "api-key": EMBEDDING_API_KEY,
                "Content-Type": "application/json"
                 }

            payload = {
                "input": texts,
                "dimensions": 256
                 }
   
            url = EMBBEDDING_URL
   
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
              data = response.json()
              embeddings = [item['embedding'] for item in data['data']]
              return list(var_ids, embeddings) 
        
            else:
                 raise Exception(f"Error: {response.status_code} - {response.text}")
            
    

        # run embeding for all batches
        batch_num = 1
        while True:
             records = fetch_batch_from_db(batch_size)
             if not records:
               logger.info(" All records have been embedded.")
               break

             logger.info(f"\n Processing batch {batch_num} with {len(records)} records...")
             results = embed_batch(records)
             save_embedding_to_db(results)
             logger.info(f"✅ Batch {batch_num} saved.")

             batch_num += 1
             time.sleep(1)  # interupt for avoiding overloading
    
     
    def search_products_by_text(query_text, top_k=5):
     EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
    
     EMBBEDDING_URL = os.getenv("EMBEDDING_URL")
     
     def get_query_embedding(text):
        headers = {
           "api-key":  EMBEDDING_API_KEY,
           "Content-Type": "application/json"
        }
        payload = {
           "input": [text],
           "dimensions": 256
         }
        url =  EMBBEDDING_URL
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
          data = response.json()
          return data['data'][0]['embedding']
        else:
          raise Exception(f"Embedding API error: {response.status_code} - {response.text}")

     embedding = get_query_embedding(query_text)
     conn =  psycopg2.connect(
            host=os.getenv("DB_HOST")   ,     
            port= os.getenv("DB_PORT")  ,   
            database= os.getenv("DB_NAME") ,   
            user=os.getenv("DB_USER")     ,   
            password= os.getenv("DB_PASSWORD")  
        )
     cursor = conn.cursor()
     sql = """
        SELECT p.id, p.name, p.price, p.category, pd.variation,
               1 - (pe.embedding <#> %s::vector) AS similarity
        FROM product_embeddings pe
        JOIN product_descriptions pd ON pe.variation_id = pd.id
        JOIN products p ON pd.product_id = p.id
        ORDER BY pe.embedding <#> %s::vector
        LIMIT %s;
     """

     # pgvector wants the embedding as a string like: '[0.1, 0.2, ...]'
     emb_str = "[" + ", ".join(map(str, embedding)) + "]"
     cursor.execute(sql, (emb_str, emb_str, top_k))
     results = cursor.fetchall()

     conn.close()
     return results

                
# Example usage
if __name__ == "__main__":
    processor = woocomerce_ext()
    # processor.fetch_site_data()
    # processor.put_db_products("./product.json")
    processor.generate_variation_text()
    # processor.compute_embedding(5) # 5 is the size of batch
    # processor.search_products_by_text("فیلم برداری حرفه ای")
      
