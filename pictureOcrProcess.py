
import asyncio
import base64
import logging
import os
import random

from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeDocumentRequest,
    AnalyzeResult,
    ContentFormat,
)
from azure.ai.vision.imageanalysis.aio import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv(verbose=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

endpoint = os.getenv("FORM_RECOGNIZER_ENDPOINT")
key = os.getenv("FORM_RECOGNIZER_KEY")


cvEndpointList = [
    {"cvEndpoint": os.getenv("AZURE_COMPUTER_VISION_ENDPOINT1"), "cvEndpointKey": os.getenv("AZURE_COMPUTER_VISION_KEY1")},
    {"cvEndpoint": os.getenv("AZURE_COMPUTER_VISION_ENDPOINT2"), "cvEndpointKey": os.getenv("AZURE_COMPUTER_VISION_KEY2")},
    {"cvEndpoint": os.getenv("AZURE_COMPUTER_VISION_ENDPOINT3"), "cvEndpointKey": os.getenv("AZURE_COMPUTER_VISION_KEY3")}
]

# 当前使用的索引，初始为0
endpoint_index = 0
endpoint_lock = asyncio.Lock()  # 用于保护 endpoint_index 的锁

async def analyze_document(document_path: str):
    logging.info(f"Analyzing document {document_path}")

    async with DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key)) as document_analysis_client:
        poller = await document_analysis_client.begin_analyze_document(
                "prebuilt-layout", 
                AnalyzeDocumentRequest(bytes_source= await convert_pdf_to_base64(document_path)),
                output_content_format=ContentFormat.MARKDOWN
            )
        result: AnalyzeResult  = await poller.result()
        return result.content

async def convert_pdf_to_base64(pdf_path: str):
    logging.info(f"Converting PDF to base64: {pdf_path}")
    # Read the PDF file in binary mode, encode it to base64, and decode to string
    with open(pdf_path, "rb") as file:
        base64_encoded_pdf = base64.b64encode(file.read()).decode()
    return base64_encoded_pdf


async def get_image_caption_byCV(image_url: str, max_retries=5) -> str:
    logging.info(f"Getting caption of image {image_url}")
    
    retry_count = 0
    backoff_time = 1  # 初始退避时间为 1 秒
    global endpoint_index

    while retry_count < max_retries:
        try:
             # 使用锁来保护对 endpoint_index 的访问
            async with endpoint_lock:
                cvEndpoint = cvEndpointList[endpoint_index]
                endpoint_index = (endpoint_index + 1) % len(cvEndpointList)  # 轮询下一个端点

            # 创建 ImageAnalysisClient 实例
            async with ImageAnalysisClient(endpoint=cvEndpoint['cvEndpoint'], credential=AzureKeyCredential(cvEndpoint["cvEndpointKey"])) as imageAnalysisClient:
                result = await imageAnalysisClient.analyze_from_url(
                    image_url=image_url,
                    visual_features=[VisualFeatures.CAPTION, VisualFeatures.READ, VisualFeatures.DENSE_CAPTIONS],
                    gender_neutral_caption=False
                )

            # 处理返回的 dense captions
            if result.dense_captions["values"] is not None:
                values_list = result.dense_captions["values"]
                combined_text = ''.join(item['text'] for item in values_list)
                return combined_text
            else:
                return ""

        except Exception as e:
            if "429" in str(e):
                # 捕获限流错误，使用指数退避重试
                logging.warning(f"Rate limit exceeded. Retrying in {backoff_time} seconds...")
                await asyncio.sleep(backoff_time + random.uniform(0, 0.5))  # 增加随机抖动
                retry_count += 1
                backoff_time *= 2  # 每次重试后退避时间加倍
            else:
                # 对于非 429 错误，直接抛出异常
                logging.error(f"Failed to get caption for image {image_url}: {e}")
                raise

    raise Exception(f"Exceeded maximum retries ({max_retries}) for image {image_url}")

if __name__ == "__main__":
    # 示例调用
    # document_path = "docs/pdf/lnXUR7aSAmIIRZsSITN9BFxmou0f.pdf"
    # result = asyncio.run(analyze_document(document_path))
    # print("picture's ocr content: {}",result)

    image_url="https://img2.tapimg.com/moment/etag/FqoXHRQGKEuYj-ViJ-FTcPXHkRbs.png"
    caption = asyncio.run(get_image_caption_byCV(image_url,5))
    print("image caption: {}",caption)

