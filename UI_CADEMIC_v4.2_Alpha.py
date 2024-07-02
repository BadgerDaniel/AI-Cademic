import os
import re
import pandas as pd
import numpy as np
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from pydantic import FieldSerializationInfo
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.output_parsers import RegexParser, PydanticOutputParser
from pprint import pprint
import random
import warnings
warnings.filterwarnings(action="ignore")
import requests
import PyPDF2
import dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel
from pinecone import Pinecone
import torch
import ftfy
from tqdm import tqdm
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
# from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz
from PIL import Image
import io
from openai import OpenAI
import pdfplumber
import json
from langchain.chains import ConversationalRetrievalChain
import streamlit as st

warnings.filterwarnings(action="ignore")

# Setting up environment variables
openai_api_key = st.secrets["openai"]["api_key"]
pinecone_api_key = st.secrets["pinecone"]["api_key"]
pinecone_api_env = st.secrets["pinecone"]["api_env"]

client = OpenAI(api_key=openai_api_key)

pc = Pinecone(api_key=pinecone_api_key)
# pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))


myindex=pc.Index('ai-cademic')

with open (r'dicembed.json', 'r') as file:
    dic_embed=json.load(file)


def get_embedding(text, model="text-embedding-3-small"):
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def get_llm_answer(question, context, api_key):
    # Validate input parameters
    if not api_key:
        raise ValueError("API key is required")
    if not question:
        raise ValueError("Question is required")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": question + context
            }
        ],
        "max_tokens": 300
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad HTTP status codes

        # Extract the plain text answer from the response
        answer = response.json()['choices'][0]['message']['content'].strip()
        return answer

    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


#get_llm_answer(question=user_question+instructions,context=answer_context,api_key=openai_api_key)



# Instructions for context
instructions = " Here's some context, only use it if it is relevant to answer the question, if it is not, mention that the information found did not satisfy the need of the user. The user who asked the preceding question likely has not seen this context, so adjust your answer accordingly."

# Function to process text
def process_text(input_text):
    queryresult = myindex.query(
        vector=get_embedding(input_text),
        top_k=5,
        include_values=True
    )
    #answer_context = dic_embed[queryresult['matches'][0]['id']]
    
    answer_context=' '
    for x in range(0,min(len(queryresult),1):
        answer_context += str(dic_embed[queryresult['matches'][x]['id']])


    return get_llm_answer(question=input_text + instructions, context=answer_context, queryresult=queryresult)

# Streamlit App
def main():
    st.title("AI-CADEMIC")

    # Initialize session state for conversation
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # Multi-line prompt and multi-line text input
    prompt = """Hi, I'm AI-CADEMIC. 
    Your go-to assistant for querying academic articles on generative AI. 
    
    Ask me a question about generative AI.
    I'll provide you with insights from relevant academic materials :)
    """

    #Whether you're seeking information on attention mechanisms, challenges in large language models, or multimodal AI, I've got you covered!
    
    user_input = st.text_area(prompt)

    # Button to submit text
    if st.button("Submit"):
        with st.spinner("Racking my brain.."):
            response=process_text(user_input)

        # display response
        st.write('Response: ',response)
            

if __name__ == '__main__':
    main()
