from supabase import create_client, Client
from sentence_transformers import SentenceTransformer, util
import re
import os
from openai import AzureOpenAI
import json
from logger import logger
import db_init
from converting import convert_time_standard, convert_price_standard
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI


def searchByFilter(query_text, filter1,top_k=5):# can add 3 other filter in arguman


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


 class Info(BaseModel):
    subject: str
    time: str = Field(description="زمان ذکرشده در متن (مثلاً: 'کمتر از سه ساعت یا 120 دقیقه')")
    price: str = Field(description="قیمت ذکرشده در متن (مثلاً: '۲۵۰ هزار تومان')")

 # 2. ساخت parser
 parser = PydanticOutputParser(pydantic_object=Info)
 prompt = PromptTemplate(
    template="""
    متن زیر را بخوان و اطلاعات خواسته شده را استخراج کن:
    متن: {text}
    
    {format_instructions}
    """,
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
 )

 model = ChatOpenAI(temperature=0)

 _input = prompt.format(text=query_text)
 output = model.predict(_input)

 parsed = parser.parse(output)
 print("subject:", parsed.subject)
 print("زمان:", parsed.time)
 print("قیمت:", parsed.price)

 price= convert_price_standard(parsed.price)
 time= convert_time_standard(parsed.time)

 embed_subject = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[parsed.subject],
        dimensions=DIMENSIONS,
        timeout= 60
        )

 embedding = embed_subject.data[0].embedding
 emb_subject = "[" + ", ".join(map(str, embedding)) + "]"

 query = """
    SELECT 
     p.product_id
    FROM 
      product_descriptions AS p
    INNER JOIN 
     (
        SELECT 
            variation_id, 
            embedding <-> %s AS distance
        FROM 
            product_embeddings
        ORDER BY 
            distance
        LIMIT %s
     ) AS e
    ON 
     p.id = e.variation_id;
"""

 cursor.execute(query, (emb_subject, top_k))

 filtered_id = cursor.fetchall()
 product_ids = [row[0] for row in filtered_id]


 logger.info(f" filtered variation id: {product_ids}")

# 
 query = f"SELECT id, name, price, {filter1} FROM products WHERE id = ANY (%s)"

 params = []
 params.append(product_ids)

 if  price is not None:
    query += f" AND price < %s"
    params.append(price)

 if  time is not None:
    query += f" AND {filter1} < %s"
    params.append(time)

# if parsed.size is not None:
#     query += f" AND {filter2} = %s"
#     params.append(parsed.size)

# if parsed.text is not None:
#     query += f" AND {filter3} like %s"
#     params.append(parsed.text)

# if parsed.date is not None:
#     query += f" AND {filter4} < %s"
#     params.append(parsed.date)

 query += "LIMIT %s"
 params.append(top_k)


#  find_product = f"SELECT 
#     id, 
#     name,
#     price, { filter1 } , {filter2} ,{filter3} ,{filter4}
#     FROM products where id in (%s)
#     and (price < %s or %s is null)
#     and (filter1 < %s or %s is null)
#     and (filter2 < %s or %sis null)
#     and (filter3 < %s or %s is null)
#     and (filter4 < %s or %s is null)    
#     limit %s
#    "
 cursor.execute(query,tuple(params))
 results = cursor.fetchall()
 cursor.close()

 
 

 logger.info(f"Search for '{query_text}' returned {len(results)} results.")
 return [
        {"name": r[1], "price": r[2], "course time": r[3]}
        for r in results
    ] 




