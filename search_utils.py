import asyncio
import logging
import os
from typing import List

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType, VectorizedQuery
from dotenv import load_dotenv
from openai import AzureOpenAI

from multiModelsEmbedding import (
    get_picture_embedding,
    get_text_embedding_by_computer_vision,
)
from pictureFormatProcess import download_and_save_as_pdf
from pictureOcrProcess import analyze_document, get_image_caption_byCV

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure environment variables  
load_dotenv()  
azure_search_service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT") 
azure_search_index_name = os.getenv("AZURE_SEARCH_INDEX") 
azure_search_key = os.getenv("AZURE_COGNITIVE_SEARCH_KEY") 
azure_search_credential = AzureKeyCredential(azure_search_key)


azureOpenAIClient = AzureOpenAI(
  api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version = "2024-02-01",
  azure_endpoint =os.getenv("AZURE_OPENAI_BASE") 
)

azure_openAI_embedding_deployment = os.getenv("EMBEDDING_MODEL_DEPLOYMENT")
azure_computer_vision_endpoint = os.getenv("AZURE_COMPUTER_VISION_ENDPOINT")
azure_computer_vision_key = os.getenv("AZURE_COMPUTER_VISION_KEY")
pdf_dir = "docs/pdf"


async def get_search_results_by_image(query_image_url:str):
     # generate ocr content by form recognizer service
    pdfFileLocalPath =  await download_and_save_as_pdf(query_image_url,pdf_dir)
    ocrContent = await analyze_document(pdfFileLocalPath)
    captionByCV = await get_image_caption_byCV(query_image_url)

    query = ocrContent + captionByCV
    
    aoaiResponse = azureOpenAIClient.embeddings.create(input = query,model = azure_openAI_embedding_deployment)  
    aoai_embedding_query = aoaiResponse.data[0].embedding
    #print(aoai_embedding_query)

    cv_embedding_query = await get_picture_embedding(query_image_url)
    #print(cv_embedding_query)

    search_client = SearchClient(azure_search_service_endpoint, azure_search_index_name, AzureKeyCredential(azure_search_key))

    aoai_embedding_query = VectorizedQuery(vector=aoai_embedding_query, 
                                k_nearest_neighbors=3, 
                                fields="contentVector,captionVector,ocrContentVecotor")

    azure_cv_embedding_query = VectorizedQuery(vector=cv_embedding_query, 
                                k_nearest_neighbors=3, 
                                fields="imageVecotor")

    results = search_client.search(  
        search_text=query,
        search_fields=["caption","content","ocrContent"],
        query_language="zh-cn",
        scoring_profile="firstProfile",   
        vector_queries=[aoai_embedding_query,azure_cv_embedding_query],
        query_type=QueryType.SEMANTIC, 
        semantic_configuration_name='default', 
        select=["id","caption", "content","imageUrl","ocrContent"],
        top=3
    )

    return results

async def get_search_results_by_text(query_text:str):
    aoaiResponse = azureOpenAIClient.embeddings.create(input = query_text,model = azure_openAI_embedding_deployment)  
    aoai_embedding_query = aoaiResponse.data[0].embedding
    cv_embedding_query = await get_text_embedding_by_computer_vision(query_text)

    search_client = SearchClient(azure_search_service_endpoint, azure_search_index_name, AzureKeyCredential(azure_search_key))

    aoai_embedding_query = VectorizedQuery(vector=aoai_embedding_query, 
                                k_nearest_neighbors=3, 
                                fields="contentVector,captionVector,ocrContentVecotor")

    azure_cv_embedding_query = VectorizedQuery(vector=cv_embedding_query, 
                                k_nearest_neighbors=3, 
                                fields="imageVecotor")

    results = search_client.search(  
        search_text=query_text,
        search_fields=["caption","content","ocrContent"],
        query_language="zh-cn",
        scoring_profile="firstProfile",   
        vector_queries=[aoai_embedding_query,azure_cv_embedding_query],
        query_type=QueryType.SEMANTIC, 
        semantic_configuration_name='default', 
        select=["id","caption", "content","imageUrl","ocrContent"],
        top=3
    )

    return results

async def get_search_results_by_image_and_text(query_image_url:str,query_text:str):
    aoaiResponse = azureOpenAIClient.embeddings.create(input = query_text,model = azure_openAI_embedding_deployment)  
    aoai_embedding_query = aoaiResponse.data[0].embedding
    #print(aoai_embedding_query)

    cv_embedding_query = await get_picture_embedding(query_image_url)
    #print(cv_embedding_query)

    search_client = SearchClient(azure_search_service_endpoint, azure_search_index_name, AzureKeyCredential(azure_search_key))

    aoai_embedding_query = VectorizedQuery(vector=aoai_embedding_query, 
                                k_nearest_neighbors=3, 
                                fields="contentVector,captionVector,ocrContentVecotor")

    azure_cv_embedding_query = VectorizedQuery(vector=cv_embedding_query, 
                                k_nearest_neighbors=3, 
                                fields="imageVecotor")

    results = search_client.search(  
        search_text=query_text,
        search_fields=["caption","content","ocrContent"],
        query_language="zh-cn",
        scoring_profile="firstProfile",   
        vector_queries=[aoai_embedding_query,azure_cv_embedding_query],
        query_type=QueryType.SEMANTIC, 
        semantic_configuration_name='default', 
        select=["id","caption", "content","imageUrl","ocrContent"],
        top=3
    )

    return results

if __name__ == "__main__":

    query_image_url="https://img2.tapimg.com/moment/etag/FvhNYMQT78nnCjAvBqHvY40FcH46.jpeg"

    query = "DNF手游伤害为什么是黄字？"

    results = asyncio.run(get_search_results_by_image_and_text(query_image_url,query))
    print("####################Results####################")
    
    for result in results:
        print(f"Reranker Score: {result['@search.reranker_score']}")
        print(f"Score: {result['@search.score']}")  
        print(f"Captions: {result['@search.captions']}")  
        print(f"Highlights: {result['@search.highlights']}")  
        print(f"caption: {result['caption']}\n")  
        print(f"content: {result['content']}\n")  
        print(f"ocrContent: {result['ocrContent']}\n")  
        print(f"imageUrl: {result['imageUrl']}\n")  
        print("###############################")
