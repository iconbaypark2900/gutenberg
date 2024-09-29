import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
    MONGODB_URI = os.getenv('MONGODB_URI')
    ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD')
    
    @staticmethod
    def get(key, default=None):
        return getattr(Config, key, default)