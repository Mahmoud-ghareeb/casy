"""Loads json documents."""
import json
from abc import ABC
from typing import List
from urllib.parse import urlparse

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class JsonCourseLoader(BaseLoader, ABC):
    def __init__(self, file_path: str):
        """Initialize with file path."""
        self.file_path = file_path
        with open(file_path, encoding="utf-8") as f:
            self.data = json.load(f)

    def load(self) -> List[Document]:
        documents = []
        for video in self.data:
            name = 'الحلقة ' + str(video['videoId'])
            title = 'العنوان ' + video['video_title']
            body = 'المحتوي '
            for i in video['paragraphInfo']:
                body += i['paragraphDetails']

            full_data = f'{name} {title} {body}'

            documents.append(Document(
                page_content=full_data,
                metadata={"source": self.file_path},
            ))

        return documents