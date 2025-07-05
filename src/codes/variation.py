import os
import json
import yaml
import time
from tqdm import tqdm
import concurrent.futures
from openai import AzureOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
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
        num_of_var = config["inputParameter"]["num_of_variation"]

    client = AzureOpenAI(
        api_key=API_KEY,
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT
    )
    
    # class VariationsSchema(BaseModel):
    #     variations: list[str] = Field(..., description="A list of short, complete product description variations.")

    variations_schema = ResponseSchema(name="variations", description="A list of short, complete product description variations.")
    output_parser = StructuredOutputParser(response_schemas=[variations_schema])

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
        for future in tqdm(concurrent.futures.as_completed(future_to_product), total=len(product_tuples), desc="Generating variations"):
            product_id, variations = future.result()
            for variation in variations:
                if variation:
                    updates.append((product_id, variation))

    # with get_conn() as conn:
    #     cursor = conn.cursor()
    #     cursor.executemany(
    #         """INSERT INTO product_descriptions (product_id, variation)
    #             VALUES (%s, %s)""",
    #         updates
    #     )
    #     conn.commit()
    #     cursor.close()
    logger.info("All products processed.") 