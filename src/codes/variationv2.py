import os
import json
import yaml
import time
from tqdm import tqdm
import concurrent.futures
from openai import AzureOpenAI
from pydantic.v1 import BaseModel, Field
from langchain.output_parsers.structured import StructuredOutputParser
from langchain.output_parsers import ResponseSchema
from db_init import get_conn
from logger import logger

def generate_variation_text():
    API_KEY = os.getenv("API_KEY")
    API_VERSION = os.getenv("API_VERSION")
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
    MODEL_NAME = "gpt-4o-mini"
    SLEEP_TIME = 5
    MAX_WORKERS = 3
    with open("config.yaml", "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
        summerization_prompt = config["inputParameter"]["summerization_prompt"]
        key_points_prompt = config["inputParameter"]["key_points_prompt"]

    client = AzureOpenAI(
        api_key=API_KEY,
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT
    )
    
   

    #variations_schema = ResponseSchema(name="variations", description="A list of short, complete product description variations.")
   # output_parser = StructuredOutputParser(response_schemas=[variations_schema])



    rquest_summerize = ''' "{summerization}" این توضیحات محصول است: "{description}"  '''
    request_key_points = ''' "{key_points}" این توضیحات محصول است: "{description}"  '''
    
    def generate_variations_for_product(product):
        product_id, description = product
        try:
            prompt1 = request_key_points.format(description=description, key_points=key_points_prompt)
            prompt2 = rquest_summerize.format(description=description, summerization=summerization_prompt)
           
            key_points_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt1}],
                max_tokens=1024,
                temperature=0.7,
            )
            keys = key_points_response.choices[0].message.content
            
            summary_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt2}],
                max_tokens=1024,
                temperature=0.7,
            )
           
            summary = summary_response.choices[0].message.content
            
            time.sleep(SLEEP_TIME)
            return (product_id, summary, keys)
        except Exception as e:
            logger.error(f"Error for product {product_id}: {str(e)}")
            return (product_id, "", "")

    # Fetch all products first
    with get_conn() as conn:
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

    updates = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_product = {executor.submit(generate_variations_for_product, product): product for product in product_tuples}
        for future in tqdm(concurrent.futures.as_completed(future_to_product), total=len(product_tuples), desc="Generating summaries and key points"):
            product_id, summary, key_points = future.result()
            if summary:
                updates.append((product_id, summary))
            if key_points:
                updates.append((product_id, key_points))

    with get_conn() as conn:
         cursor = conn.cursor()
         cursor.executemany(
             """INSERT INTO product_descriptions (product_id, variation)
                 VALUES (%s, %s)""",
             updates
         )
         conn.commit()
         cursor.close()
    logger.info("All products processed.") 