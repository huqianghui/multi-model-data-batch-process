"""Data utilities for index preparation."""
import asyncio

from multiModelsEmbedding import get_picture_embedding
from multiModelsPictureProcess import get_content_by_mulit_model
from objectDefinition import Document, ImageData, RecordResult
from pictureFormatProcess import download_and_save_as_pdf
from pictureOcrProcess import analyze_document, get_image_caption_byCV
from textEmbeddingProcess import get_text_embedding

pdf_dir = "docs/pdf"

async def process_images_records(file_path: str)->RecordResult:
    
    documents = []
    errorRecords = []
    image_data_list = []

    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    # Manually parse the line to extract fields
                    line = line.strip()
                    id_start = line.find("'id': '") + len("'id': '")
                    id_end = line.find("',", id_start)
                    image_id = line[id_start:id_end]

                    image_url_start = line.find("'imageUrl': '") + len("'imageUrl': '")
                    image_url_end = line.find("',", image_url_start)
                    image_url = line[image_url_start:image_url_end]

                    caption_start = line.find("'caption': '") + len("'caption': '")
                    caption_end = line.rfind("'}")
                    caption = line[caption_start:caption_end]

                    # Escape special characters
                    image_id = image_id.replace("'", "\\'")
                    image_url = image_url.replace("'", "\\'")
                    caption = caption.replace("'", "\\'")

                    # Create an ImageData object
                    image_data = ImageData(id=image_id, imageUrl=image_url, caption=caption)
                    # Add the object to the list
                    image_data_list.append(image_data)
                except Exception as e:
                    print(f"Error processing line: {line}")
                    print(f"Error message: {e}")
    except Exception as e:
        print(f"Error processing file: {file_path}")
        raise e
    
    recordResult = RecordResult(documentList=documents, failedImageList=errorRecords, totalRecords=len(image_data_list))

    for item in image_data_list:
        try:
            id = item.id
            url = item.imageUrl
            caption = item.caption

            # create async tasks 
            content_task = asyncio.create_task(get_content_by_mulit_model(url))
            pdf_task = asyncio.create_task(download_and_save_as_pdf(url, pdf_dir))
            caption_task = asyncio.create_task(get_image_caption_byCV(url))
            image_embedding_task = asyncio.create_task(get_picture_embedding(url))

            # wait for all tasks to complete
            content = await content_task
            pdfFileLocalPath = await pdf_task
            captionByCV = await caption_task
            imageVector = await image_embedding_task

            # generate OCR content
            ocrContent = await analyze_document(pdfFileLocalPath)

            # get text embeddings task
            captionVector_task = asyncio.create_task(get_text_embedding(captionByCV))
            contentVector_task = asyncio.create_task(get_text_embedding(content))
            ocrContentVector_task = asyncio.create_task(get_text_embedding(ocrContent + captionByCV))

            # wait for all tasks to complete
            captionVector = await captionVector_task
            contentVector = await contentVector_task
            ocrContentVector = await ocrContentVector_task
            
            # create a Document object and add it to the list
            document = Document( id=id, 
                                imageUrl=url, 
                                caption=caption, 
                                content=content, 
                                ocrContent=ocrContent, 
                                captionVector=captionVector, 
                                contentVector=contentVector, 
                                ocrContentVecotor=ocrContentVector, 
                                imageVecotor=imageVector)
        
            documents.append(document)
        except Exception as e:
            print(f"Error processing record: {item.id}")
            print(f"Error message: {e}")
            errorRecords.append(item)

    return recordResult

if __name__ == "__main__":
    # 示例调用
    recordResult = asyncio.run(process_images_records("multi-models/image_captions/ima_files_2_test.txt"))    
    print("recordResult: {}",recordResult)