from sentence_transformers import SentenceTransformer
from utils import error_handler, log_function_call

model = SentenceTransformer('all-MiniLM-L6-v2')

@error_handler
@log_function_call
def interpret_query(query):
    # Generate query embedding
    query_embedding = model.encode(query).tolist()
    
    return {
        "original_query": query,
        "embedding": query_embedding
    }