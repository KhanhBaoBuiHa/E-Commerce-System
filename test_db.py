import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))
df = pd.read_sql("SELECT * FROM user_last_viewed_product LIMIT 5", engine)
print(df)