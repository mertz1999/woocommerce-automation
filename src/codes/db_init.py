import os
import psycopg2
from psycopg2.extras import execute_values

def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

def create_tables():
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
               id INTEGER PRIMARY KEY,
               name TEXT,
               price TEXT,
               meta_data JSONB,
               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
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
                variation_id INT NOT NULL REFERENCES product_descriptions(id) UNIQUE,
                embedding vector(256),
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        cursor.close() 