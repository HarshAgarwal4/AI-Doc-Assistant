import os
from pathlib import Path

from dotenv import load_dotenv
from sqlmodel import Session, create_engine

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
PRODUCTION = os.getenv("PRODUCTION")

if not DATABASE_URL:
    raise RuntimeError("DATABASE URL IS MISSING")

engine = create_engine(DATABASE_URL, echo=PRODUCTION != "True")

def get_session():
    with Session(engine) as session:
        yield session
