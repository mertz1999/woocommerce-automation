
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

class woocomerce_ext(API):
    def __init__(self, httpsurl, wckey, wcsecret, db_path="woocomerce.db" ):
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

    def fetch_site_data(self):  #getting all products data from site(name- description- image- price -...)
        
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
        print("fetching compelete")
    
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
      print("data inserted to database")

    def generate_variation_text(self):
       
        # SET API KEY
        OPENROUTER_API_KEY = "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"
        MODEL_NAME = "meta-llama/llama-3.1-8b-instruct:free"

        #  تنظیمات کلی
        MAX_WORKERS = 5  # تعداد تردهای همزمان
        SLEEP_TIME = 1.5  # زمان مکث بین درخواست‌ها

          
        llm = ChatOpenAI(
         model_name=MODEL_NAME,
         openai_api_base="https://openrouter.ai/api/v1",
         openai_api_key=OPENROUTER_API_KEY,
        )

        # قالب پرامپت برای تولید نسخه‌های متفاوت از توضیح محصول
        prompt = PromptTemplate.from_template(
            """این یک توضیح محصول است:
                        "{description}"
                            لطفاً {n} نسخه متفاوت و ترغیب‌کننده از این متن برای فروشگاه اینترنتی بنویس. لحنت باید دوستانه و قانع‌کننده باشد."""
                        )

        # خروجی را به شکل متن ساده دریافت می‌کنیم
        chain = prompt | llm | StrOutputParser()

        def generate_variations( description, n=2):
          try:
           output = chain.invoke({"description": description, "n": n})
           print(output)
           return output
          except Exception as e:
           raise Exception(f"LangChain Error: {str(e)}")

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
             print(f"✅ ذخیره شد: محصول {product_id}")
             time.sleep(SLEEP_TIME)

           except Exception as e:
             print(f"❌ خطا برای محصول {product_id}: {str(e)}")

         #  سیستم صف برای مدیریت چندتردی
        def worker():
         while True:
           item = q.get()
           if item is None:
             break
           product_id, description = item
           process_product(product_id, description)
           q.task_done()

    #  خواندن محصولات از دیتابیس
    
         self.cursor.execute("SELECT id, description FROM products")
         products = self.cursor.fetchall()
         self.conn.close()

        #  ایجاد تردها و صف
         q = Queue()
         threads = []

         for i in range(MAX_WORKERS):
          t = threading.Thread(target=worker)
          t.start()
          threads.append(t)

#  افزودن کارها به صف
         for product in products:
          q.put(product)

#  منتظر بمان تا همه کارها تمام شوند
         q.join()

#  بستن تردها
         for i in range(MAX_WORKERS):
           q.put(None)
           for t in threads:
             t.join()

         print("\n🎉 تمام محصولات پردازش شدند.")


   


    def compute_embedding(self):


        #  تنظیمات اولیه
        OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"
        MODEL_NAME = "openai/text-embedding-ada-002"
        NUM_THREADS = 4

       
        #  آماده‌سازی جدول خروجی
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
            """تولید embedding برای متن و ذخیره آن در دیتابیس."""
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
                return f"✅ ذخیره شد: id={id} "
            except Exception as e:
                return f"❌ خطا: {str(e)}"

        #  سیستم Thread و Queue
        q = Queue()

        def worker(agent):
            while True:
                item = q.get()
                if item is None:
                    break
                row_id, des, var = item
                agent.invoke({"input": f"ذخیره embedding  برای متن اول و دوم:\n{des}\n{var}", "id": row_id, "description": "des", "variations": var})
               
                q.task_done()
                time.sleep(1)

        # 📋 خواندن داده‌ها
        
        self.cursor.execute("SELECT id, text1, text2 FROM text_data")
        rows = self.cursor.fetchall()
        self.conn.close()

        # 🧠 Agent را بساز
        tools = [embed_and_store]
        agent = initialize_agent(
            tools=tools,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

        # 🧵 راه‌اندازی تردها
        threads = []
        for _ in range(NUM_THREADS):
            t = threading.Thread(target=worker, args=(agent,))
            t.start()
            threads.append(t)

        # ➕ پر کردن صف
        for row in rows:
            q.put(row)

        q.join()

        # 🛑 بستن تردها
        for _ in range(NUM_THREADS):
            q.put(None)
        for t in threads:
            t.join()

        print("\n🎉 همه embedding ها انجام و ذخیره شدند.")

   

# نمونه استفاده
if __name__ == "__main__":
    processor = woocomerce_ext(

"https://motiongraphistan.com", "ck_2425836dfb4a29fda20f49981a350bd1d3237042", "cs_285c3d6d0704e85cb53972996d64a5e5fadb4970", db_path="woocomerce.db" )

   # processor.fetch_site_data()
  #  processor.put_db_products()
    processor.generate_variation_text()
      
