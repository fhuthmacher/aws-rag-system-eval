# load environment variables 
import boto3
import os
import botocore
from botocore.config import Config
import langchain
import sagemaker
import pandas as pd

from langchain.llms.bedrock import Bedrock
from langchain.llms import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from typing import Dict

import json
import requests
import csv
import time
import pandas as pd
import nltk
import sys

from langchain.llms import Bedrock
from dotenv import load_dotenv, find_dotenv
import mlflow
from mlflow import MlflowClient

import random
random_identifier = random.randint(100,999)

# loading environment variables that are stored in local file dev.env
local_env_filename = 'dev-rageval.env'
load_dotenv(find_dotenv(local_env_filename),override=True)

os.environ['OPENSEARCH_COLLECTION'] = os.getenv('OPENSEARCH_COLLECTION')
os.environ['BEDROCK_KNOWLEDGEBASE_ID'] = os.getenv('BEDROCK_KNOWLEDGEBASE_ID')
os.environ['REGION'] = os.getenv('REGION')
os.environ['MLFLOW_TRACKING_ENABLED'] = os.getenv('MLFLOW_TRACKING_ENABLED')
os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI')

if os.environ['MLFLOW_TRACKING_ENABLED'] == 'True':
   # Initialize mlflow client
   mlflow_client = MlflowClient(tracking_uri=os.environ['MLFLOW_TRACKING_URI'])


# Initialize Bedrock runtime
config = Config(
   retries = {
      'max_attempts': 10,
      'mode': 'standard'
   }
)
bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        config=config
)
bedrock_client = boto3.client(
        service_name="bedrock",
        config=config
)

# Initialize sagemaker session
session = sagemaker.Session()
bucket = session.default_bucket()


## 1. Download ground truth dataset

import xmltodict
url = 'https://d3q8adh3y5sxpk.cloudfront.net/rageval/qsdata_20.xml'

# Send an HTTP GET request to download the file
response = requests.get(url)

# Check if the request was successful (HTTP status code 200)
if response.status_code == 200:        
    xml_data = xmltodict.parse(response.text)

# Convert the dictionary to a Pandas DataFrame
qa_dataset = pd.DataFrame(xml_data['data']['records'])

prompts = []
for row in qa_dataset.itertuples():
    item = {
        'prompt': str(row[1]['Question']),
        'context': str(row[1]['Context']),
        'output': '<question_answer>' + str(row[1]['Answer']['question_answer']) + '</question_answer>',
        'page': str(row[1]['Page'])
    }
    prompts.append(item)

# example prompt
print(prompts[0])

# 2. Download context / Amazon annual report and create documents
import numpy as np
import pypdf
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from urllib.request import urlretrieve

os.makedirs("data", exist_ok=True)
files = [ "https://d3q8adh3y5sxpk.cloudfront.net/rageval/AMZN-2023-10k.pdf"]
for url in files:
    file_path = os.path.join("data", url.rpartition("/")[2])
    urlretrieve(url, file_path)
    

loader = PyPDFDirectoryLoader("./data/")
documents = loader.load()

prompt_template_claude_1 = """
        Human: 
        You are a helpful, respectful, and honest research assistant, dedicated to providing valuable and accurate information.
        You will be provided with a report extract between <report></report> XML tags, please read it and analyse the content.
        Please answer the following question: 
        {question} 
        
        The answer must only be based on the information from the report.
        Return the answer inside <question_answer></question_answer> XML tags.

        If a particular bit of information is not present, return "There is not enough information available to answer this question" inside the XML tags.
        Each returned answer should be concise, remove extra information if possible.
        The report will be given between <report></report> XML tags.

        <report>
        {context}
        </report>

        Return the answer inside <question_answer></question_answer> XML tags.
        Assistant:"""

# 5a. RAG Agent Helper Class
from anthropic import Anthropic
import time
from typing import List
from abc import ABC, abstractmethod
import boto3
import json
import os
from dataclasses import dataclass
import tiktoken 
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
import numpy as np
from enum import Enum
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.utils.math import cosine_similarity

class DistanceStrategy(str, Enum):
    '''
    Enumerator of the Distance strategies for calculating distances between vectors.
    '''
    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
    MAX_INNER_PRODUCT = "MAX_INNER_PRODUCT"
    DOT_PRODUCT = "DOT_PRODUCT"
    JACCARD = "JACCARD"
    COSINE = "COSINE"

@dataclass
class RagAnswer():
    '''
    RAG answer structure
    '''
    answer: str
    context_list: List[str]
    query_time: float
    usage: int
    cost: float

class RagAgent(ABC):
    def __init__(
        self,
        model_id: str = 'anthropic.claude-3-haiku-20240307-v1:0',
        embedding_model_id: str = 'amazon.titan-embed-text-v1',
        top_k: int = 20,
        top_p: float = 0.7,
        temperature: float = 0.0,
        max_token_count: int = 4000,
        anthropic_version: str = 'bedrock-2023-05-31',
        prompt_template: str = '',
        search_method: str = 'approximate_search', # 'approximate_search' / 'mmr' / 'bedrock_kb'
        knowledge_base_id: str = '',
        region: str = 'us-east-1',
        opensearch_endpoint: str = '',
        opensearch_index: str = 'rag_eval',
        opensearch_index_dimension: int = 1536,
        opensearch_fetch_k: int = 4, # OpenSearch default
        documents: [] = [],
        chunking_strategy: str = 'TokenTextSplitter', # RecursiveCharacterTextSplitter / TokenTextSplitter
        chunk_size: int = 2000, 
        chunk_overlap: int = 260,
        debug: bool = False

    ):
        self.model_id = model_id
        self.embedding_model_id = embedding_model_id
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.max_token_count = max_token_count
        self.anthropic_version = anthropic_version
        self.prompt_template = prompt_template
        self.search_method = search_method
        self.knowledge_base_id = knowledge_base_id
        self.region = region
        self.opensearch_endpoint = opensearch_endpoint
        self.opensearch_index = opensearch_index
        self.opensearch_index_dimension = opensearch_index_dimension
        self.opensearch_fetch_k = opensearch_fetch_k
        self.documents = documents
        self.chunking_strategy = chunking_strategy
        self.chunk_size =  chunk_size
        self.chunk_overlap = chunk_overlap
        self.debug = debug

        self.bedrock_agent_runtime = boto3.client(
            service_name="bedrock-agent-runtime"
        )
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime"
        )
        self.anthropic_client = Anthropic()

        credentials = boto3.Session().get_credentials()
        auth = AWSV4SignerAuth(credentials, self.region, 'aoss')
        self.aos_client = OpenSearch(
            hosts=[{'host': self.opensearch_endpoint, 'port': 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            pool_maxsize=20,
        )
    
    def count_tokens(text):
        '''
        using Anthropic library client to count tokens of a given text
        '''
        return self.anthropic_client.count_tokens(text)
    
    def calculate_cost(self, usage, model_id):
        '''
        Takes the usage tokens returned by Bedrock in input and output, and coverts to cost in dollars.
        '''
        input_token_haiku = 0.25/1000000
        output_token_haiku = 1.25/1000000
        input_token_sonnet = 3.00/1000000
        output_token_sonnet = 15.00/1000000
        input_token_opus = 15.00/1000000
        output_token_opus = 75.00/1000000
        cost = 0

        if 'haiku' in model_id:
            cost+= usage['input_tokens']*input_token_haiku
            cost+= usage['output_tokens']*output_token_haiku
        if 'sonnet' in model_id:
            cost+= usage['input_tokens']*input_token_sonnet
            cost+= usage['output_tokens']*output_token_sonnet
        if 'opus' in model_id:
            cost+= usage['input_tokens']*input_token_opus
            cost+= usage['output_tokens']*output_token_opus

        return cost
    
    def get_chunks(self, documents, chunking_strategy, chunk_size, chunk_overlap):
        '''
        using Langchain textsplitter implementations to split array of documents into chunks
        '''
        if chunking_strategy == 'RecursiveCharacterTextSplitter':
            char_text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunk_list = char_text_splitter.split_documents(self.documents)
            
        if chunking_strategy == 'TokenTextSplitter':
            token_text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunk_list = token_text_splitter.split_documents(self.documents)

        final_chunk_list = []
        for chunk in chunk_list:
            final_chunk_list.append(chunk.page_content)

        return final_chunk_list

    def get_best_practice_chunks(full_text, OVERLAP=True):
        '''
        This will take a text and return an array with sliced chunks of the text following some general best practices:
        1) Overlaping allows more cohesion between text, and should only be turned off when trying to count specific numbers and no duplicated text is a requirment.
        2) Dropping text up to the maximum context window of a model, doesn't work very well.Part of the reason for this is because no matter the input length,
           the output length is about the same. For example, if you drop in a paragraph or 10 pages, you get about a paragraph in response.
        3) To mitigate this, we use chunks that are the lesser of two values: 25% of the total token count or 2k tokens.
           We'll also overlap our chunks by about a paragraph of text or so, in order to provide continuity between chunks.
           (Logic taken from https://gist.github.com/Donavan/4fdb489a467efdc1faac0077a151407a)
        '''
        
        #Following testing, it was found that chunks should be 2000 tokens, or 25% of the doc, whichever is shorter.
        #max chunk size in tokens
        chunk_length_tokens = 2000
        #chunk length may be shortened later for shorter docs.
        
        #a paragraph is about 200 words, which is about 260 tokens on average
        #we'll overlap our chunks by a paragraph to provide cohesion.
        overlap_tokens = 260
        if not OVERLAP: overlap_tokens = 0
        
        #anything this short doesn't need to be chunked further.
        min_chunk_length = 260 + overlap_tokens*2
        
        #grab basic info about the text to be chunked.
        char_count = len(full_text)
        word_count = len(full_text.split(" "))#rough estimate
        token_count = count_tokens(full_text)
        token_per_charater = token_count/char_count

        #don't chunk tiny texts
        if token_count <= min_chunk_length:
            if self.debug: print("Text is too small to be chunked further")
            return [full_text]
        
        if self.debug:
            print ("Chunk DEBUG mode is on, information about the text and chunking will be printed out.")
            print ("Estimated character count:",char_count)
            print ("Estimated word count:",word_count)
            print ("Estimated token count:",token_count)
            print ("Estimated tokens per character:",token_per_charater)

            print("Full text tokens: ",count_tokens(full_text))
            print("How many times bigger than max context window: ",round(count_tokens(full_text)/max_token_count,2))
        
        #if the text is shorter, use smaller chunks
        if (token_count/4<chunk_length_tokens):
            overlap_tokens = int((overlap_tokens/chunk_length_tokens)*int(token_count/4))
            chunk_length_tokens = int(token_count/4)
            
            if self.debug: 
                print("Short doc detected:")
                print("New chunk length:",chunk_length_tokens)
                print("New overlap length:",overlap_tokens)
            
        #convert to characters for easy slicing using our approximate tokens per character for this text.
        overlap_chars = int(overlap_tokens/token_per_charater)
        chunk_length_chars = int(chunk_length_tokens/token_per_charater)
        
        #itterate and create the chunks from the full text.
        chunks = []
        start_chunk = 0
        end_chunk = chunk_length_chars + overlap_chars
        
        last_chunk = False
        while not last_chunk:
            #the last chunk may not be the full length.
            if(end_chunk>=char_count):
                end_chunk=char_count
                last_chunk=True
            chunks.append(full_text[start_chunk:end_chunk])
            
            #move our slice location
            if start_chunk == 0:
                start_chunk += chunk_length_chars - overlap_chars
            else:
                start_chunk += chunk_length_chars

            end_chunk = start_chunk + chunk_length_chars + 2 * overlap_chars
            
        if self.debug:
            print ("Created %s chunks."%len(chunks))
        return chunks
    
    def get_embedding(self, body):
        '''
        This function is used to generate the embeddings for a specific chunk of text
        '''
        accept = 'application/json'
        contentType = 'application/json'
    
        response = self.bedrock_runtime.invoke_model(body=body, 
                                                        modelId=self.embedding_model_id, 
                                                        accept=accept, 
                                                        contentType=contentType)
    
        response_body = json.loads(response.get('body').read())
        
        return response_body

    def index_doc(self, vectors, text):
        '''
        This function is used to ingest a single chunk/document embedding into OpenSearch Serverless
        '''       
        indexDocument = {
            "vector_field": vectors,
            "text": text

        }
        try:
            response = self.aos_client.index(
                index=self.opensearch_index,
                body=indexDocument,
                refresh=False
            )

            return response
        except Exception as e:
            print(f'error: {e}')

    def index_documents(self,documents):
        '''
        This function is used to create an OpenSearch index, and then ingest all documents
        '''  
        self.documents = documents

        # create OpenSearch index
        knn_index = {
            "settings": {
                "index.knn": True
            },
            "mappings": {
                "properties": {
                    "vector_field": {
                        "type": "knn_vector",
                        "method": {
                        "engine": "faiss",
                        "name": "hnsw",
                        "space_type": "l2"
                        },
                        "dimension": self.opensearch_index_dimension,
                        "store": True
                    },
                    "text": {
                        "type": "text",
                        "store": True
                    },
                }
            }
        }

        try:
            print('trying to create new index')
            self.aos_client.indices.delete(index=self.opensearch_index)
            self.aos_client.indices.create(index=self.opensearch_index,body=knn_index,ignore=400)
            
        except:
            print(f'Index {self.opensearch_index} not found. Creating index on OpenSearch.')
            self.aos_client.indices.create(index=self.opensearch_index,body=knn_index,ignore=400)
        
        while True:
            try:
                response = self.aos_client.indices.get(index=self.opensearch_index)
                if response[f'{self.opensearch_index}']: 
                    print(f'new index {self.opensearch_index} has been created.')
                    break
            except:
                print(f'keep waiting for new index {self.opensearch_index} creation')
                time.sleep(10)

        # generate chunks
        chunks = self.get_chunks(self.documents,self.chunking_strategy,self.chunk_size,self.chunk_overlap)
        
        # generate embeddings for each chunk of text data
        docs = []
        i = 0
        for chunk in chunks:
            i += 1
            if 'amazon' in self.embedding_model_id:
                embeddingInput = json.dumps({"inputText": chunk})
                response_body = self.get_embedding(embeddingInput)
                embeddingVectors = response_body.get('embedding')
            if 'cohere' in self.embedding_model_id:
                embeddingInput = json.dumps(
                                            {
                                                "texts": [chunk],
                                                "input_type": "search_document", # |search_query|classification|clustering
                                                "truncate": "NONE" # NONE|START|END
                                            }
                                            )
            
                response_body = self.get_embedding(embeddingInput)
                embeddingVectors = response_body['embeddings'][0]

            docs.append({"_index": self.opensearch_index, "vector_field": embeddingVectors, "text": chunk})
            
        
        from opensearchpy import helpers
        # response = helpers.bulk(self.aos_client, docs, max_retries=3)
        succeeded = []
        failed = []
        for success, item in helpers.parallel_bulk(self.aos_client, 
            actions=docs, 
            chunk_size=10, 
            raise_on_error=False,
            raise_on_exception=False,
            max_chunk_bytes=20 * 1024 * 1024,
            request_timeout=60):
            
            if success:
                succeeded.append(item)
            else:
                failed.append(item)
        if self.debug:        
            print(f'succeeded: {succeeded}, failed: {failed}')
            
        return succeeded

    def retrieve_context(self, query: str) -> List[str]:
        '''
        This function is used to retrieve document embeddings from a knowledgebase that are similar to the search query.
        ''' 
        if len(self.knowledge_base_id) > 0 and self.search_method == 'bedrock_kb':
            # print(f'Retrieving with Bedrock Knowledge Base ID: {self.knowledge_base_id}')
            retrieval_results = self.bedrock_agent_runtime.retrieve(
                retrievalQuery={"text": query},
                knowledgeBaseId=self.knowledge_base_id,
                retrievalConfiguration={
                    "vectorSearchConfiguration": {"numberOfResults": self.top_k
                                                  #"overrideSearchType": "HYBRID", # optional
                    }
                }
            )["retrievalResults"]

            return [x["content"]["text"] for x in retrieval_results]
        
        # use other retriever options, e.g. OpenSearch directly
        if self.search_method == 'approximate_search' or self.search_method == 'mmr':
            if self.debug:
                print(f'Retrieving from OpenSearch Index {self.opensearch_index} with search method {self.search_method}')
            
            # formatting the user input
            if 'amazon' in self.embedding_model_id:
                query_body = json.dumps({"inputText": query})
                response_body = self.get_embedding(query_body)
                vectors = response_body.get('embedding')
            if 'cohere' in self.embedding_model_id:
                query_body = json.dumps(
                                            {
                                                "texts":[query],
                                                "input_type": "search_query", # |search_query|classification|clustering
                                                "truncate": "NONE" # NONE|START|END
                                            }
                                        )
                response_body = self.get_embedding(query_body)
                vectors = response_body['embeddings'][0]

            # the query parameters for the KNN search 
            query = {
                "size": self.opensearch_fetch_k,
                "query": {
                    "knn": {
                        "vector_field": {
                            "vector": vectors, "k": self.opensearch_fetch_k
                        }
                    }
                },
                "_source": True,
                "fields": ["text"],
            }

            aos_response = self.aos_client.search(
                body=query,
                index=self.opensearch_index
            )
        
            results = aos_response["hits"]["hits"]

        if self.search_method == 'mmr':

            embeddings = []
            for result in results:
                embeddings.append(result["_source"]["vector_field"])

            # Rerank top k results using MMR, (mmr_selected is a list of indices)
            # lambda_mult: Number between 0 and 1 that determines the degree
            #             of diversity among the results with 0 corresponding
            #             to maximum diversity and 1 to minimum diversity.
            #             Defaults to 0.5.

            lambda_mult = 0.5
            mmr_selected = self.maximal_marginal_relevance(
                np.array(vectors), embeddings, k=self.opensearch_fetch_k, lambda_mult=lambda_mult
            )

        retrieval_results= []
        if self.search_method == 'mmr':
            for i in mmr_selected:
                retrieval_results.append(results[i]["fields"]["text"][0])

        else:
            for i in results:
                retrieval_results.append(i["fields"]["text"][0])
        
        return retrieval_results

    def maximal_marginal_relevance(self, query_embedding: np.ndarray, embedding_list: list, lambda_mult: float = 0.5, k: int = 4) -> List[int]:
        '''
        This function is used to re-rank the retrieve document embeddings from a knowledgebase by calculating maximal marginal relevance.
        ''' 
        
        if min(k, len(embedding_list)) <= 0:
            return []
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
        similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
        most_similar = int(np.argmax(similarity_to_query))
        idxs = [most_similar]
        selected = np.array([embedding_list[most_similar]])
        while len(idxs) < min(k, len(embedding_list)):
            best_score = -np.inf
            idx_to_add = -1
            similarity_to_selected = cosine_similarity(embedding_list, selected)
            for i, query_score in enumerate(similarity_to_query):
                if i in idxs:
                    continue
                redundant_score = max(similarity_to_selected[i])
                equation_score = (
                    lambda_mult * query_score - (1 - lambda_mult) * redundant_score
                )
                if equation_score > best_score:
                    best_score = equation_score
                    idx_to_add = i
            idxs.append(idx_to_add)
            selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
        return idxs

    def get_prompt(self, question, context_list):
        '''
        This function is used to generate the final prompt that is passed to the LLM to generate an answer for a given query.
        ''' 
        context = ""
        for context_chunk in context_list:
            context += context_chunk + '\n'
        final_prompt = self.prompt_template.format(question=question, context=context)
        return final_prompt

    def get_response_from_model(self, model_prompt: str,context_list:[] ) -> RagAnswer:
        '''
        This function is used to invoke an LLM that is available via Amazon Bedrock.
        ''' 
        attempt = 1
        query_time = -1
        usage = (-1,-1)
        start_time = time.time()
        if 'anthropic.claude-3' in self.model_id:
            
            MAX_ATTEMPTS = 3
            messages = model_prompt
            system = ''
            
            #if the messages are just a string, convert to the Messages API format.
            if type(messages)==str:
                messages = [{"role": "user", "content": messages}]

            #build the JSON to send to Bedrock
            prompt_json = {
                "system":system,
                "messages": messages,
                "max_tokens": self.max_token_count, # 4096 is a hard limit to output length in Claude 3
                "temperature": self.temperature, #creativity on a scale from 0-1.
                "anthropic_version": self.anthropic_version,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "stop_sequences": ["\n\nHuman:"]
            }
    
            if self.debug: 
                print("Sending:\nSystem:\n",system,"\nMessages:\n",str(messages))
    
            while True:
                try:
                    response = self.bedrock_runtime.invoke_model(body=json.dumps(prompt_json), modelId=self.model_id, accept='application/json', contentType='application/json')
                    response_body = json.loads(response.get('body').read())
                    results = response_body.get("content")[0].get("text")
                    usage = response_body.get("usage")
                    
                    query_time = round(time.time()-start_time,2)
                    if self.debug:
                        print("Retrieved:",results)
                    
                    break
                except Exception as e:
                    print("Error with calling Bedrock: "+str(e))
                    attempt+=1
                    if attempt>MAX_ATTEMPTS:
                        print("Max attempts reached!")
                        results = str(e)
                        break
                    else:
                        time.sleep(10)
            if self.debug: 
                print(f'usage: {usage} and query_time: {query_time}')

        if 'cohere' in self.model_id:
            body = {
                "prompt": model_prompt,
                "max_tokens": self.max_token_count,
                "temperature": self.temperature,
                "p": 0.7,
                "k": self.top_k,
                "num_generations": 1,
                "return_likelihoods": "GENERATION"
            }

            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id, body=json.dumps(body)
            )

            response_body = json.loads(response["body"].read())

            generations = response_body.get('generations')

            for index, generation in enumerate(generations):
                results = generation['text']
                
            usage = response_body.get("usage")
            query_time = round(time.time()-start_time,2)

        if 'titan' in self.model_id:

            body = {
                "inputText": model_prompt,
                "textGenerationConfig":
                {
                    "temperature": self.temperature,
                    "maxTokenCount": self.max_token_count,
                }
            }

            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id, body=json.dumps(body)
            )

            response_body = json.loads(response["body"].read())
            results = response_body["results"][0]["outputText"]
            usage = response_body.get("usage")
            query_time = round(time.time()-start_time,2)
            if self.debug: 
                print(f'usage: {usage} and query_time: {query_time}')

        cost = self.calculate_cost(usage, self.model_id)

        return RagAnswer(results,context_list,query_time,usage,cost)
         
        
    def search(self, question: str) -> RagAnswer:
        '''
        This function is used to search/answer a given user query with a given knowledge base.
        ''' 
        context_list = self.retrieve_context(question)
        if self.debug:
            print(f'context: {context_list}')
        
        model_prompt = self.get_prompt(question, context_list)
        if self.debug:
            print(f'model_prompt: {model_prompt}')

        response = self.get_response_from_model(model_prompt, context_list)
        if self.debug:
            print(f'model_response: {response}')

        return response

# 5b. ragas helper class to use RAGAS with Amazon Bedrock

from ragas.llms import BaseRagasLLM
from ragas.embeddings import BaseRagasEmbeddings
from ragas.run_config import RunConfig
from datasets import Dataset
import typing as t
from langchain_core.embeddings import Embeddings
from langchain_core.outputs import LLMResult
from langchain_core.callbacks import Callbacks
from langchain_core.outputs.generation import Generation
from ragas.llms.prompt import PromptValue
import time
from queue import Queue
from threading import Thread
import boto3


class BedrockLLMWrapper():
    def __init__(self,
        model_id: str = 'anthropic.claude-3-haiku-20240307-v1:0', #'anthropic.claude-3-sonnet-20240229-v1:0',
        top_k: int = 5,
        top_p: int = 0.7,
        temperature: float = 0.0,
        max_token_count: int = 4000,
        anthropic_version: str = "bedrock-2023-05-31",
        max_attempts: int = 3,
        debug: bool = False

    ):

        self.model_id = model_id
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.max_token_count = max_token_count
        self.anthropic_version = anthropic_version
        self.max_attempts = max_attempts
        self.debug = debug
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime"
        )
        
    def generate(self,prompt):
        if self.debug: 
            print('entered BedrockLLMWrapper generate')
        attempt = 1
        query_time = -1
        usage = (-1,-1)
        start_time = time.time()
        messages = prompt
        system = ''

        #if the messages are just a string, convert to the Messages API format.
        if type(messages)==str:
            messages = [{"role": "user", "content": messages}]
        
        #build the JSON to send to Bedrock
        prompt_json = {
            "system":system,
            "messages": messages,
            "max_tokens": self.max_token_count, # 4096 is a hard limit to output length in Claude 3
            "temperature": self.temperature, #creativity on a scale from 0-1.
            "anthropic_version":self.anthropic_version,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "stop_sequences": ["\n\nHuman:"]
        }

        if self.debug: 
            print("Sending:\nSystem:\n",system,"\nMessages:\n",str(messages))

        while True:
            try:
                response = self.bedrock_runtime.invoke_model(body=json.dumps(prompt_json), modelId=self.model_id, accept='application/json', contentType='application/json')
                response_body = json.loads(response.get('body').read())
                result_text = response_body.get("content")[0].get("text")
                usage = response_body.get("usage")
                query_time = round(time.time()-start_time,2)
                if self.debug:
                    print("Retrieved:",result_text)
                
                break
                
            except Exception as e:
                print("Error with calling Bedrock: "+str(e))
                attempt+=1
                if attempt>self.max_attempts:
                    print("Max attempts reached!")
                    result_text = str(e)
                    break
                else:#retry in 10 seconds
                    print("retry")
                    time.sleep(10)
        if self.debug: 
            print(f'usage: {usage} and query_time: {query_time}')

        # return result_text
        return [result_text,usage,query_time]

    # Threaded function for queue processing.
    def thread_request(self, q, result):
        while not q.empty():
            work = q.get()    #fetch new work from the Queue
            try:
                data = self.generate(work[1])
                result[work[0]] = data  #Store data back at correct index
            except Exception as e:
                print('Error with prompt!',str(e))
                result[work[0]] = (str(e))
            #signal to the queue that task has been processed
            q.task_done()
        return True

    def generate_threaded(self,prompts):
        '''
        Call ask_claude, but multi-threaded.
        Returns a dict of the prompts and responces.
        '''
        system=""
        ignore_cache=False
        q = Queue(maxsize=0)
        num_theads = min(50, len(prompts))
        #Populating Queue with tasks
        results = [{} for x in prompts];
        #load up the queue with the promts to fetch and the index for each job (as a tuple):
        for i in range(len(prompts)):
            #need the index and the url in each queue item.
            q.put((i,prompts[i]))
            
        #Starting worker threads on queue processing
        for i in range(num_theads):
            if self.debug:
                print('Starting thread ', i)
            worker = Thread(target=self.thread_request, args=(q,results))
            worker.daemon = True
            worker.start()

        #now we wait until the queue has been processed
        q.join()
        return results



class BedrockRagasLLM(BaseRagasLLM):
    def __init__(self, name,run_config: t.Optional[RunConfig] = None):
        super().__init__(name)
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

    def calculate_cost(self, usage, model_id):
        '''
        Takes the usage tokens returned by Bedrock in input and output, and coverts to cost in dollars.
        '''
        input_token_haiku = 0.25/1000000
        output_token_haiku = 1.25/1000000
        input_token_sonnet = 3.00/1000000
        output_token_sonnet = 15.00/1000000
        input_token_opus = 15.00/1000000
        output_token_opus = 75.00/1000000
        cost = 0

        if 'haiku' in model_id:
            cost+= usage['input_tokens']*input_token_haiku
            cost+= usage['output_tokens']*output_token_haiku
        if 'sonnet' in model_id:
            cost+= usage['input_tokens']*input_token_sonnet
            cost+= usage['output_tokens']*output_token_sonnet
        if 'opus' in model_id:
            cost+= usage['input_tokens']*input_token_opus
            cost+= usage['output_tokens']*output_token_opus

        return cost

    def generate_text(
            self,
            prompt,
            n: int = 1,
            temperature: float = 0.0,
            stop: t.Optional[t.List[str]] = [],
            callbacks: Callbacks = None,
        ):

        wrapper = BedrockLLMWrapper()
        result = wrapper.generate(prompt.prompt_str)
        cost = self.calculate_cost(result[1], wrapper.model_id)
        generation_info = {"input_tokens": result[1]['input_tokens'], "output_tokens": result[1]['output_tokens'], "cost": cost , "query_time": result[2] }
        generation = Generation(text=result[0], type='Generation', generation_info=generation_info)

        # Create a list to hold generations
        generations_list = []
        generation_sublist = []
        generation_sublist.append(generation)
        generations_list.append(generation_sublist)

        return LLMResult(generations=generations_list)

    async def agenerate_text(
            self,
            prompt: PromptValue,
            n: int = 1,
            temperature: float = 0,
            stop: t.Optional[t.List[str]] = [],
            callbacks: Callbacks = None,
        ) -> LLMResult:

        prompts_list = []
        prompts_list.append(prompt.prompt_str)
        wrapper = BedrockLLMWrapper()
        results = wrapper.generate_threaded(prompts_list)
        
       

        # Create a list to hold generations
        generations_list = []
        generation_sublist = []
        for result in results:
            # generation = Generation(text=text, type='Generation')
            cost = self.calculate_cost(result[1], wrapper.model_id)
            generation_info = {"input_tokens": result[1]['input_tokens'], "output_tokens": result[1]['output_tokens'], "cost": cost , "query_time": result[2] }
            generation = Generation(text=result[0], type='Generation', generation_info=generation_info)
            generation_sublist.append(generation)
        generations_list.append(generation_sublist)

        return LLMResult(generations=generations_list)

# 5c. helper class to evaluate RAG system with RAGAS

import mlflow
from mlflow.data.pandas_dataset import PandasDataset
from datasets import Dataset
import ragas
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    faithfulness,
    context_recall,
    answer_relevancy,
    answer_correctness,
    # answer_similarity,
    # context_relevancy
)
import boto3
import json
import base64
from langchain_core.globals import set_debug
set_debug(False)

class RagasEval():

    def run_ragas_eval(self, rag_system_eval_details, debug=False):
        # if you run into any error, set langchain debug to true
        if debug == True:
            set_debug(True)
        else:
            set_debug(False)

        ragas_evals_df = pd.DataFrame()
        mlflow_metrics_results = {}
        
        experiment_name = rag_system_eval_details["experiment_name"]
        rag_agent = rag_system_eval_details["rag_agent"]
        ground_truth = rag_system_eval_details["ground_truth"]
        
        random_identifier = random.randrange(100, 1000, 3)
        run_name=f'LLM_{rag_agent.model_id}_embeddings{rag_agent.embedding_model_id}_searchmethod_{rag_agent.search_method}_template_{rag_agent.prompt_template}_chunking_strategy_{rag_agent.chunking_strategy}_chunk_size_{rag_agent.chunk_size}_chunk_overlap_{rag_agent.chunk_overlap}_{random_identifier}'    

        metrics = [
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
            answer_correctness,
            # answer_similarity,
            # context_relevancy
            
        ]

        basic_qa_ragas_dataset = []
        basic_qa_details = []
        basic_error_log = []

        for item in ground_truth:
            generated_answer_error = ""
            retrieved_context_error = ""
            reference_answer_error = ""
            
            # RagAnswer(results,context_list,query_time,usage,cost)
            result = rag_agent.search(item['prompt'])

            question = item['prompt']
            reference_answer = item['output']
            generated_answer = result.answer
            retrieved_context = result.context_list
            context_sequence = []

            

            for doc in result.context_list:
                context_sequence.append(doc)
            retrieved_context = context_sequence

            if generated_answer is None or generated_answer.strip() == "" or generated_answer.replace(" ", "") == "<question_answer></question_answer>":
                generated_answer_error = f'no answer generated for question {question}'
                
            if retrieved_context is None or retrieved_context == "" or retrieved_context == "[]" or len(context_sequence) == 0 :
                retrieved_context_error = f'no retrieved_context for question {question}'
            
            if reference_answer is None or reference_answer == "":
                reference_answer_error = f'no reference_answer for question {question}'
                    
            
            if generated_answer_error != "" or retrieved_context_error != "" or reference_answer_error != "":
                basic_error_log.append(
                    {"question" : question,
                    "generated_answer_error" : generated_answer_error,
                    "retrieved_context_error" : retrieved_context_error,
                    "reference_answer_error" : reference_answer_error,
                    "value_error": ""
                    }
                )
            else:
                
                basic_qa_ragas_dataset.append(
                    {"question" : item['prompt'],
                    "answer" : result.answer,
                    "contexts" : context_sequence,
                    "ground_truth" : item['output']
                    }
                )
                basic_qa_details.append({
                    "query_time": result.query_time,
                    "usage": result.usage,
                    "cost": result.cost
                }
                )

        
        #capture any errors in CSV for further review
        if len(basic_error_log) != 0:
            basic_error_log_df = pd.DataFrame(basic_error_log)
            basic_error_log_df.to_csv('./mlflow_run_errorlog.csv',index=False)
        
        if len(basic_qa_ragas_dataset) != 0:
            basic_qa_ragas_df = pd.DataFrame(basic_qa_ragas_dataset)
            basic_qa_ragas_df.to_csv('./mlflow_run_predictions.csv',index=False)
            basic_qa_details_df = pd.DataFrame(basic_qa_details)

            # evaluate with RAGAS
            basic_qa_ragas = Dataset.from_pandas(basic_qa_ragas_df)
            
            ragas_eval_llm = BedrockRagasLLM('BaseRagasLLM')
            from langchain.embeddings import BedrockEmbeddings
            ragas_eval_embedding = BedrockEmbeddings(
                client=bedrock_runtime,
                model_id="amazon.titan-embed-text-v1"
            )

            # evaluate with RAGAS
            basic_qa_ragas = Dataset.from_pandas(basic_qa_ragas_df)
            run_config=RunConfig(timeout=15, max_retries=1, max_wait=15, max_workers=6)
            ragas_result = evaluate(basic_qa_ragas, metrics=metrics, llm=ragas_eval_llm, embeddings=ragas_eval_embedding,run_config=run_config, is_async=True,raise_exceptions=True )
            

            ragas_evals_df = ragas_result.to_pandas()

            params = {
                "run_name": run_name,
                "llm_model_id": rag_agent.model_id,
                "embedding_model_id": rag_agent.embedding_model_id,
                "knowledge_base_id": rag_agent.knowledge_base_id,
                "chunking_strategy": rag_agent.chunking_strategy,
                "chunk_size": rag_agent.chunk_size,
                "chunk_overlap": rag_agent.chunk_overlap,
                "prompt_template": rag_agent.prompt_template,
                "opensearch_index": rag_agent.opensearch_index,
                "opensearch_index_dimension": rag_agent.opensearch_index_dimension,
                "opensearch_fetch_k": rag_agent.opensearch_fetch_k,
                "search_method": rag_agent.search_method

            }
            print(f'RAG system mean query time: {basic_qa_details_df["query_time"].mean()}')
            print(f'RAG system cost on ground truth set: {basic_qa_details_df["cost"].sum()}')
            print(f'ragas_faithfulness mean: {ragas_evals_df["faithfulness"].mean()}')
            print(f'ragas_answer_relevancy mean: {ragas_evals_df["answer_relevancy"].mean()}')
            print(f'ragas_context_recall mean:: {ragas_evals_df["context_recall"].mean()}')
            print(f'ragas_context_precision mean: {ragas_evals_df["context_precision"].mean()}')
            # print(f'ragas_context_relevancy mean: {ragas_evals_df["context_relevancy"].mean()}')
            # print(f'ragas_answer_similarity mean: {ragas_evals_df["answer_similarity"].mean()}')
            print(f'ragas_answer_correctness mean: {ragas_evals_df["answer_correctness"].mean()}')

            mlflow_metrics_results = {
                "ragas_faithfulness_mean": ragas_evals_df["faithfulness"].mean(),
                "ragas_answer_relevancy_mean": ragas_evals_df["answer_relevancy"].mean(),
                "ragas_context_recall_mean": ragas_evals_df["context_recall"].mean(),
                "ragas_context_precision_mean": ragas_evals_df["context_precision"].mean(),
                # "ragas_context_relevancy_mean": ragas_evals_df["context_relevancy"].mean(),
                # "ragas_answer_similarity_mean": ragas_evals_df["answer_similarity"].mean(),
                "ragas_answer_correctness_mean": ragas_evals_df["answer_correctness"].mean(),

            }

            if os.environ['MLFLOW_TRACKING_ENABLED'] == 'True':
                llm_experiment = mlflow.set_experiment(experiment_name)
                # Initiate the MLflow run context
                with mlflow.start_run(run_name=run_name) as run:
                    # Log input dataset to MLflow Tracking as a JSON artifact.
                    mlflow_dataset = mlflow.data.from_pandas(basic_qa_ragas_df, source='./mlflow_run_predictions.csv')
                    mlflow.log_input(mlflow_dataset, context="rag-output")

                    # Log evaluation metrics that were calculated
                    mlflow.log_metrics(mlflow_metrics_results)
                    
                    # Log RAGAS results to MLflow Tracking as a JSON artifact.
                    mlflow.log_table(data=ragas_evals_df, artifact_file="qa_ragas_eval_results.json")

                    # Log parameters used for the RAG system
                    mlflow.log_params(params)

        return mlflow_metrics_results, ragas_evals_df
    
    def analze_ragas_result_chart(self, image_path, comparison):
        with open(image_path, "rb") as image_file:
            image_b64_string = base64.b64encode(image_file.read()).decode("utf-8")

        # Variables for Bedrock API
        modelId = 'anthropic.claude-3-haiku-20240307-v1:0' #'anthropic.claude-3-sonnet-20240229-v1:0'
        contentType = 'application/json'
        accept = 'application/json'
        prompt = """The chart shows a comparison of the mean RAGAS metrics for faithfulness, answere relevance, answer correctness, context recall, and context precision. Follow the below instructions.
                    1. Analyse the chart carefully.
                    2. Determine which option performs best if the metrics are prioritized in the followind order: 
                    1)context recall,  2)context precision, 3)faithfulness, 4)answer relevance, 5)answer correctness. 
                    Include the result in the response in <best_prioritized></best_prioritized> xml tags. 
                    3. Determine a use case and reason why someone might want to proceed with this option and why this option performed better compared to the other options ({comparison}). Include the reason in no more than 150 words in the response <best_prioritized_reason></best_prioritized_reason> xml tags."""

        # Messages
        messages = [
        {
            "role": "user",
            "content": [
            {
                "type": "image",
                "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image_b64_string
                }
            },
            {
                "type": "text",
                "text": prompt
            }
            ]
        }
        ]

        # Body
        body = json.dumps({
            "anthropic_version": 'bedrock-2023-05-31',
            "max_tokens": 1000,
            "messages": messages
        })

        # Run Bedrock API
        response = bedrock_runtime.invoke_model(
        modelId=modelId,
        contentType=contentType,
        accept=accept,
        body=body
        )

        response_body = json.loads(response.get('body').read())

        return response_body['content'][0]['text']

# 8.1.1.1 Chunk Size Eval with chunk_size=128, chunk_overlap=64

rag_agent = RagAgent(   model_id = 'anthropic.claude-3-sonnet-20240229-v1:0',
                        embedding_model_id = 'amazon.titan-embed-text-v1',
                        top_k = 20,
                        top_p = 0.7,
                        temperature = 0.0,
                        max_token_count = 4000,
                        prompt_template = prompt_template_claude_1,
                        search_method = 'approximate_search', # 'approximate_search' / 'mmr' / 'bedrock_kb'
                        knowledge_base_id = os.environ['BEDROCK_KNOWLEDGEBASE_ID'],
                        region = os.environ['REGION'],
                        opensearch_endpoint = os.environ['OPENSEARCH_COLLECTION'],
                        opensearch_index = 'rag-eval-tokentextsplitter128',
                        opensearch_index_dimension = 1536,
                        opensearch_fetch_k = 4, # OpenSearch default
                        documents = documents,
                        chunking_strategy = 'TokenTextSplitter',
                        chunk_size =  128,
                        chunk_overlap = 64,
                        debug = False 
                    )

# rag_agent.index_documents(documents)
# # wait to have bulk insert complete
# time.sleep(60)
experiment_name = 'no mlflow'
ragas_eval = RagasEval()
rag_system_eval_details = {
    "rag_agent": rag_agent,
    "ground_truth": prompts,
    "experiment_name": experiment_name
}

ragas_eval_resultchunk1,ragas_eval_resultchunk1df = ragas_eval.run_ragas_eval(rag_system_eval_details)