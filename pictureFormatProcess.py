import asyncio
import logging
import os
from io import BytesIO

import httpx
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def download_image(image_url: str) -> Image.Image:
    logging.info(f"Downloading image from {image_url}")
    
    async with httpx.AsyncClient() as client:
        response = await client.get(image_url)
        response.raise_for_status()  # 如果请求失败，则引发异常
        image = Image.open(BytesIO(response.content))
        return image

async def save_image_as_pdf(image: Image.Image, pdf_path: str):
    logging.info(f"Saving image as PDF to {pdf_path}")

    pdf_bytes = BytesIO()
    image.save(pdf_bytes, format="PDF")
    pdf_bytes.seek(0)
    
    # 将 PDF 字节流保存为文件
    with open(pdf_path, "wb") as pdf_file:
        pdf_file.write(pdf_bytes.getvalue())

async def download_and_save_as_pdf(image_url: str, pdf_dir: str) -> str:
    logging.info(f"Downloading image from {image_url} and saving as PDF to {pdf_dir}")

    image = await download_image(image_url)
    image_name = os.path.basename(image_url)
    pdf_name = os.path.splitext(image_name)[0] + ".pdf"
    pdf_path = os.path.join(pdf_dir, pdf_name)
    await save_image_as_pdf(image, pdf_path)
    return pdf_path

if __name__ == "__main__":
    # 示例调用
    image_url = "https://img2.tapimg.com/moment/etag/FqoXHRQGKEuYj-ViJ-FTcPXHkRbs.png"
    pdf_dir = "docs/pdf"
    result = asyncio.run(download_and_save_as_pdf(image_url, pdf_dir))
    print("pdf file path: ", result)