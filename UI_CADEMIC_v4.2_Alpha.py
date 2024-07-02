import os
import re
import pandas as pd
import numpy as np
import requests
import json
import warnings
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.output_parsers import RegexParser, PydanticOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone

warnings.filterwarnings(action="ignore")

# Setting up environment variables
openai_api_key = st.secrets["openai"]["api_key"]
pinecone_api_key = st.secrets["pinecone"]["api_key"]
pinecone_api_env = st.secrets["pinecone"]["api_env"]

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)
myindex = pc.Index('ai-cademic')

# Load dictionary embeddings
with open(r'dicembed.json', 'r') as file:
    dic_embed = json.load(file)

# Function to get embeddings
def get_embedding(text, model="text-embedding-ada-002"):
    response = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        },
        json={
            "input": text,
            "model": model
        }
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

# Function to get LLM answer
def get_llm_answer(question, context, queryresult):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
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

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()

    answer_context = ""
    for x in range(0, 1): 
        j = dic_embed[queryresult['matches'][x]['id']]
        answer_context += j + '\n'

    return answer_context

# Instructions for context
instructions = " Here's some context, only use it if it is relevant to answer the question, if it is not, mention that the information found did not satisfy the need of the user. The user who asked the preceding question likely has not seen this context, so adjust your answer accordingly."

# Function to process text
def process_text(input_text):
    queryresult = myindex.query(
        vector=get_embedding(input_text),
        top_k=5,
        include_values=True
    )
    answer_context = dic_embed[queryresult['matches'][0]['id']]

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
        if user_input:
            # Append user input to conversation
            st.session_state.conversation.append(("User", user_input))

            with st.spinner("Racking my brain.."):
                response = process_text(user_input)

            # Append AI response to conversation
            st.session_state.conversation.append(("AI-CADEMIC", response))

    # Display the conversation in chat-like format
    for speaker, text in st.session_state.conversation:
        st.write(f"**{speaker}:** {text}")

if __name__ == '__main__':
    main()
