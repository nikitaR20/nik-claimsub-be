# app/ai/utils.py
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)

def load_table(table_name: str) -> pd.DataFrame:
    """
    Load a table from the database into a DataFrame.
    """
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT * FROM {table_name}"))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df
