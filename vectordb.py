import os
import openai
from llama_index.core import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.file import MarkdownReader
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

markdown_reader = MarkdownReader()
documents = markdown_reader.load_data("./document.md") # loads a single document into a list of documents
index = GPTVectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("Which new product and services were announced in the first quarter of 2022?")
print(response)