from utils import error_handler, log_function_call
from config import Config
import openai

openai.api_key = Config.OPENAI_API_KEY

@error_handler
@log_function_call
def generate_response(relevant_chunks, query):
    context = "\n".join([chunk['text'] for chunk in relevant_chunks])
    
    prompt = f"""
    Based on the following context and query, generate a comprehensive and coherent response:

    Context:
    {context}

    Query: {query}

    Response:
    """

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0].text.strip()