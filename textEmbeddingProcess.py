import asyncio
import logging
import os

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

load_dotenv(verbose=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


azureOpenAIClient = AsyncAzureOpenAI(
  api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version = "2024-02-01",
  azure_endpoint =os.getenv("AZURE_OPENAI_ENDPOINT") 
)

embedding_deployment = os.getenv("EMBEDDING_MODEL_DEPLOYMENT")


async def get_text_embedding(text):
    logging.info(f"Getting text embedding for {text}")
    
    response = await azureOpenAIClient.embeddings.create(input = text,model = embedding_deployment)
    return response.data[0].embedding

if __name__ == "__main__":
    # 示例调用
    input = "hello world!"
    result = asyncio.run(get_text_embedding(input))
    print(result)