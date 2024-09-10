
import asyncio
import base64
import logging
import os

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

cvEndpoint = os.getenv("AZURE_COMPUTER_VISION_ENDPOINT")
cvEndpointKey = os.getenv("AZURE_COMPUTER_VISION_KEY")


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


async def get_image_caption_byCV(image_url: str) -> str:

    logging.info(f"Getting caption of image {image_url}")
    async with ImageAnalysisClient(endpoint=cvEndpoint, credential=AzureKeyCredential(cvEndpointKey)) as imageAnalysisClient:
        result = await imageAnalysisClient.analyze_from_url(
            image_url=image_url,
            visual_features=[VisualFeatures.CAPTION, VisualFeatures.READ, VisualFeatures.DENSE_CAPTIONS],
            gender_neutral_caption=False
        )
        
    if result.dense_captions["values"] is not None:
        values_list = result.dense_captions["values"]
        combined_text = ''.join(item['text'] for item in values_list)
        return combined_text
    
    else:
        return ""

if __name__ == "__main__":
    # 示例调用
    # document_path = "docs/pdf/lnXUR7aSAmIIRZsSITN9BFxmou0f.pdf"
    # result = asyncio.run(analyze_document(document_path))
    # print("picture's ocr content: {}",result)

    image_url="https://img2.tapimg.com/moment/etag/FqoXHRQGKEuYj-ViJ-FTcPXHkRbs.png"
    caption = asyncio.run(get_image_caption_byCV(image_url))
    print("image caption: {}",caption)

