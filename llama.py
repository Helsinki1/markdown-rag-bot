import os
import openai
from llama_index.core import GPTVectorStoreIndex, Settings, PromptTemplate
from llama_index.readers.file import MarkdownReader
from evaluate import evaluateResponse
import asyncio

from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

markdown_reader = MarkdownReader()
documents = markdown_reader.load_data("./document.md") # loads a single document into a list of documents
index = GPTVectorStoreIndex.from_documents(documents) # store Apple's financial info as vector embeddings

query_engine = index.as_query_engine()

test_queries = [
    "Which new product and services were announced in the first quarter of 2022?",
    "How much money did Apple spend on Research and Development in 2022?",
    "What percentage of Apple's revenue came from international markets?",
    "What was Apple's earnings per share for 2022?",
    "Describe the major lawsuits and legal issues (if any) Apple is facing",
    "What proportion of Apple's revenue originated from its services?",
    "Were there any major risks mentioned about Apple's supply chain?",
    "Did Apple experience any significant changes in leadership?",
    "Which quarter (Q1 Q2 Q3 or Q4) did Apple experience the most growth?",
    "How many workers did Apple employ in 2022? How many were full time?"
]

correct_ans = [ # answers from Google Gemini's NotebookLM
    "Updated MacBook Pro 14” and MacBook Pro 16”, powered by the Apple M1 Pro or M1 Max chip, and the third generation of AirPods",
    "$26.251 billion",
    "China accounted for $74.200 billion, and other countries accounted for $172.269 billion. More than half of Apple's revenue came from international markets.",
    "Basic earnings per share were $6.15, and diluted earnings per share were $6.11",
    "Epic Games filed a lawsuit against Apple alleging antitrust violations related to the App Store.",
    "Services net sales accounted for $78.129 billion - roughly 20% of Apple's revenue came from its services.",
    "The COVID-19 pandemic caused disruptions to the supply chain, resulting in component shortages that affected sales worldwide",
    "There is no explicit mention of significant leadership changes in the filing.",
    "The document indicates that Apple has historically experienced higher net sales in its first quarter compared to other quarters due to seasonal holiday demand",
    "The document does not specify the exact number of employees or the number of full-time employees Apple had in 2022.",
]


prompt_template = PromptTemplate(
    """
        You are a financial assistant designed to extract data from U.S. Security and Exchange Commission files.
        Your responses must solely rely on the information provided: {context}
        The question you must answer: {query}
    """)


for i in range(10):
    context = query_engine.query(test_queries[i]) # CONTEXT queried from vector embeddings
    context_ls = []

    if context is not None:
        context_ls.append(context.response)
    else:
        context_ls.append("No context was found for the query.")

    response = Settings.llm.predict( # RESPONSE generated from LLM + context
        prompt = prompt_template,
        context = context,
        query = test_queries[i],
        max_new_tokens = 100,
    )
        
    print("LLM Response:", response)
    print("Context: ", context)

    asyncio.run( 
        evaluateResponse(
            test_queries[i],
            context_ls,
            response,
            correct_ans[i],
        )
    )

# print(query_engine.query(test_queries[0]))