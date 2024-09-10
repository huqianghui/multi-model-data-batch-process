import argparse
import asyncio
import os
import time

from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureDeveloperCliCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    AIServicesVisionParameters,
    AIServicesVisionVectorizer,
    AzureOpenAIParameters,
    AzureOpenAIVectorizer,
    CorsOptions,
    HnswAlgorithmConfiguration,
    ScoringProfile,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    TextWeights,
    VectorSearch,
    VectorSearchProfile,
)
from dotenv import load_dotenv
from tqdm import tqdm

# 加载 .env 文件中的环境变量
load_dotenv()


def create_search_index(index_name, index_client):
    print(f"Ensuring search index {index_name} exists")
    if index_name not in index_client.list_index_names():
        scoring_profile = ScoringProfile(
            name="firstProfile",
            text_weights=TextWeights(weights={
                                        "caption": 5, 
                                        "content": 1, 
                                        "ocrContent": 2}),
            function_aggregation="sum")
        scoring_profiles = []
        scoring_profiles.append(scoring_profile)
        
        index = SearchIndex(
            name=index_name,
            cors_options = CorsOptions(allowed_origins=["*"], max_age_in_seconds=600),
            scoring_profiles = scoring_profiles,
            fields=[
                SimpleField(name="id", type=SearchFieldDataType.String, key=True,searchable=False, filterable=True, sortable=True, facetable=True),
                SearchableField(name="caption", type=SearchFieldDataType.String, analyzer_name="zh-Hans.microsoft"),# caption of the picture
                SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="zh-Hans.microsoft"),# context of the picture from gpt-4o
                SimpleField(name="imageUrl", type=SearchFieldDataType.String,Searchable=False,filterable=False, sortable=True, facetable=True),# url of the picture
                SearchableField(name="ocrContent", type=SearchFieldDataType.String,analyzer_name="zh-Hans.microsoft"), # context of the picture from document intelligence
                SearchField(name="captionVector", 
                            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            hidden=False, 
                            searchable=True, 
                            filterable=False, 
                            sortable=False, 
                            facetable=False,
                            vector_search_dimensions=1536, 
                            vector_search_profile_name="azureOpenAIHnswProfile"), #  the caption's vector of the picture
                SearchField(name="contentVector", 
                            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            hidden=False, 
                            searchable=True, 
                            filterable=False, 
                            sortable=False, 
                            facetable=False,
                            vector_search_dimensions=1536, 
                            vector_search_profile_name="azureOpenAIHnswProfile"),  # content vector of the picture from gpt-4o
                SearchField(name="ocrContentVecotor", 
                            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            hidden=False, 
                            searchable=True, 
                            filterable=False, 
                            sortable=False, 
                            facetable=False,
                            vector_search_dimensions=1536, 
                            vector_search_profile_name="azureOpenAIHnswProfile"),  # content vector of the picture from document intelligence
                SearchField(name="imageVecotor", 
                            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            hidden=False, 
                            searchable=True, 
                            filterable=False, 
                            sortable=False, 
                            facetable=False,
                            vector_search_dimensions=1024, 
                            vector_search_profile_name="azureComputerVisionHnswProfile")  # content vector of the picture from computer vision
            ],
            semantic_search=SemanticSearch(
                configurations=[
                    SemanticConfiguration(
                        name="default",
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="caption"),
                            content_fields=[
                                SemanticField(field_name="content")
                            ],
                            keywords_fields=[
                                SemanticField(field_name="ocrContent")
                            ]
                        ),
                    )
                ]
            ),
            vector_search=VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="myHnsw")
                ],
                profiles=[
                    VectorSearchProfile(
                        name="azureOpenAIHnswProfile",
                        algorithm_configuration_name="myHnsw",
                        vectorizer="azureOpenAIVectorizer"),
                    VectorSearchProfile(
                        name="azureComputerVisionHnswProfile",
                        algorithm_configuration_name="myHnsw",
                        vectorizer="azureComputerVisionVectorizer")],
                vectorizers=[
                    AzureOpenAIVectorizer(
                        name="azureOpenAIVectorizer",
                        azure_open_ai_parameters=AzureOpenAIParameters(
                            resource_uri=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                            deployment_id="text-embedding-ada-002",
                            model_name="text-embedding-ada-002",
                            api_key=os.environ.get("AZURE_OPENAI_API_KEY"))),
                    AIServicesVisionVectorizer(
                        name="azureComputerVisionVectorizer",
                        ai_services_vision_parameters=AIServicesVisionParameters(
                            resource_uri=os.environ.get("AZURE_COMPUTER_VISION_ENDPOINT"),
                            api_key=os.environ.get("AZURE_COMPUTER_VISION_KEY"),
                            model_version="2023-04-15"))
                ]
            )
        )
        print(f"Creating {index_name} search index")
        index_client.create_index(index)
    else:
        print(f"Search index {index_name} already exists")


def validate_index(index_name, index_client):
    for retry_count in range(3):
        stats = index_client.get_index_statistics(index_name)
        num_chunks = stats["document_count"]
        if num_chunks == 0 and retry_count < 4:
            print("Index is empty. Waiting 60 seconds to check again...")
            time.sleep(10)
        elif num_chunks == 0 and retry_count == 4:
            print("Index is empty. Please investigate and re-index.")
        else:
            print(f"The index contains {num_chunks} chunks.")
            average_chunk_size = stats["storage_size"] / num_chunks
            print(f"The average chunk size of the index is {average_chunk_size} bytes.")
            break

async def create_and_populate_index(index_name:str, index_client:SearchIndexClient,search_client:SearchClient):
    # create or update search index with compatible schema
    create_search_index(index_name, index_client)

    file_path = os.getenv("multi_models_file_path")
    temp_dir=os.getenv("temp_dir")
    lines_per_chunk = int(os.getenv("lines_per_chunk"))
    # process data file
    small_files = await split_file(file_path, temp_dir, lines_per_chunk)

    print("chunk large file into smaller files's count:",len(small_files))
    print("Validating index...")
    validate_index(index_name, index_client)

# Function to split the large file into smaller chunks
async def split_file(file_path, temp_dir, lines_per_chunk=100):
    # Get the base file name without extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    small_files = []
    with open(file_path, 'r') as f:
        chunk = []
        file_count = 0
        for line in f:
            chunk.append(line)
            # Once we have enough lines, write them to a new small file
            if len(chunk) == lines_per_chunk:
                # Include the original file name in the chunk file name
                small_file_path = os.path.join(temp_dir, f'{file_name}_chunk_{file_count}.txt')
                with open(small_file_path, 'w') as small_file:
                    small_file.writelines(chunk)
                small_files.append(small_file_path)
                chunk = []
                file_count += 1
        
        # Write any remaining lines as the last chunk
        if chunk:
            small_file_path = os.path.join(temp_dir, f'{file_name}_chunk_{file_count}.txt')
            with open(small_file_path, 'w') as small_file:
                small_file.writelines(chunk)
            small_files.append(small_file_path)
    
    return small_files

if __name__ == "__main__":
    # Use the current user identity to connect to Azure services unless a key is explicitly set for any of them
    search_creds = AzureKeyCredential(os.getenv("AZURE_COGNITIVE_SEARCH_KEY"))
    search_service = os.getenv("AZURE_SEARCH_SERVICE")
    index_name = os.getenv("AZURE_SEARCH_INDEX")

    print("Data preparation script started")
    print("Preparing data for index:", index_name)
    search_endpoint = f"https://{search_service}.search.windows.net/"
    index_client = SearchIndexClient(endpoint=search_endpoint, credential=search_creds)

    search_client = SearchClient(
        endpoint=search_endpoint, credential=search_creds, index_name=index_name
    )

    asyncio.run(create_and_populate_index(index_name, index_client,search_client))
    print("Data preparation for index", index_name, "completed")
