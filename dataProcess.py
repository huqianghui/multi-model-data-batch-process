import argparse
import asyncio
import dataclasses
import os
import sys

from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureDeveloperCliCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from dotenv import load_dotenv
from tqdm import tqdm

# 加载 .env 文件中的环境变量
load_dotenv()

from data_utils import process_images_records


async def process_data_file(file_path:str,search_client:SearchClient):
    recordResult = await process_images_records(file_path= file_path)

    if len(recordResult.documentList) == 0:
        raise Exception("No records found. Please check the data path and records.")

    print(f"Processed file {file_path}")
    print(f"Processed {recordResult.totalRecords} records")
    print(f"records with errors: {len(recordResult.failedImageList)} records")
    print(f"valid records: {len(recordResult.documentList)} documents")

    # upload documents to index
    print("Uploading documents to index...")
    await upload_documents_to_index(recordResult.documentList, search_client)

    return recordResult

async def upload_documents_to_index(docs, search_client:SearchClient, upload_batch_size=50):
    to_upload_dicts = []

    for document in docs:
        d = dataclasses.asdict(document)
        # add id to documents
        d.update({"@search.action": "upload", "id": str(d["id"])})

        if "captionVector" in d and d["captionVector"] is None:
            del d["captionVector"]
        if "contentVector" in d and d["contentVector"] is None:
            del d["contentVector"]
        if "ocrContentVecotor" in d and d["ocrContentVecotor"] is None:
            del d["ocrContentVecotor"]
        if "imageVecotor" in d and d["imageVecotor"] is None:
            del d["imageVecotor"]

        to_upload_dicts.append(d)

    # Upload the documents in batches of upload_batch_size
    for i in tqdm(
        range(0, len(to_upload_dicts), upload_batch_size), desc="Indexing Chunks..."
    ):
        batch = to_upload_dicts[i : i + upload_batch_size]
        results =  await search_client.upload_documents(documents=batch)
        num_failures = 0
        errors = set()
        for result in results:
            if not result.succeeded:
                print(
                    f"Indexing Failed for {result.key} with ERROR: {result.error_message}"
                )
                num_failures += 1
                errors.add(result.error_message)
        if num_failures > 0:
            raise Exception(
                f"INDEXING FAILED for {num_failures} documents. Please recreate the index."
                f"To Debug: PLEASE CHECK chunk_size and upload_batch_size. \n Error Messages: {list(errors)}"
            )

if __name__ == "__main__":
    file_path = sys.argv[1]  # 获取文件路径参数
    search_creds = AzureKeyCredential(os.getenv("AZURE_COGNITIVE_SEARCH_KEY"))
    searchservice = os.getenv("AZURE_SEARCH_SERVICE")
    index_name = os.getenv("AZURE_SEARCH_INDEX")
    print("Data preparation script started")
    print("Preparing data for index:", os.getenv("AZURE_SEARCH_INDEX"))
    search_endpoint = f"https://{searchservice}.search.windows.net/"
    index_client = SearchIndexClient(endpoint=search_endpoint, credential=search_creds)

    search_client = SearchClient(endpoint=search_endpoint, credential=search_creds, index_name=index_name)

    asyncio.run(process_data_file(file_path, search_client))
    print("Data preparation for index", index_name, "completed")
