from dataclasses import dataclass
from typing import List


@dataclass
class Document:
    id: str
    imageUrl: str
    caption: str
    content: str
    ocrContent: str
    captionVector: List[float]
    contentVector: List[float]
    ocrContentVecotor: List[float]
    imageVecotor: List[float]

@dataclass
class ImageData:
    id: str
    imageUrl: str
    caption: str

@dataclass
class RecordResult:
    documentList: List[Document]
    failedImageList: List[ImageData]
    totalRecords: int