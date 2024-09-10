import asyncio
import logging
import os
from typing import List

import aiohttp
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv(verbose=True)

endpoint = os.getenv("AZURE_COMPUTER_VISION_ENDPOINT")
key = os.getenv("AZURE_COMPUTER_VISION_KEY")

async def get_picture_embedding(image_file_url:str) ->  List[float]:
    logging.info(f"Getting picture embedding for {image_file_url}")

    url = endpoint + "computervision/retrieval:vectorizeImage?api-version=2024-02-01&model-version=2023-04-15"
    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": key
    }
    body = {
        "url": image_file_url
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=body) as response:
            if response.status == 200:
                data = await response.json()
                return data['vector']
            else:
                error_text = await response.text()
                logging.error(f"Error getting picture embedding: {response.status} - {error_text}")
                raise Exception(f"Error getting picture embedding: {response.status} - {error_text}")
                

async def get_text_embedding_by_computer_vision(text:str)->  List[float]:
    logging.info(f"Getting text embedding for {text}")
    
    url = endpoint + "computervision/retrieval:vectorizeText?api-version=2024-02-01&model-version=2023-04-15"
    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": key
    }
    body = {
        "text": text
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=body) as response:
            if response.status == 200:
                data = await response.json()
                return data['vector']
            else:
                error_text = await response.text()
                logging.error(f"Error getting picture embedding: {response.status} - {error_text}")
                raise Exception(f"Error getting text embedding: {response.status} - {error_text}")

if __name__ == "__main__":
    # 示例调用
    textEmbeddingResult = asyncio.run(get_text_embedding_by_computer_vision("hello world!"))
    print("textEmbeddingResult: {}",textEmbeddingResult)

    pictureEmbeddingResult = asyncio.run(get_picture_embedding("https://img2.tapimg.com/moment/etag/lhZEbeJKeI5qOwQxlRSUTsZcYen0.png"))
    print("pictureEmbeddingResult: {}",pictureEmbeddingResult)