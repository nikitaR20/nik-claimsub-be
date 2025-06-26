from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Dummy DB connection string — update later
DATABASE_URL = "postgresql://postgres:postgres@localhost/claims_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()
