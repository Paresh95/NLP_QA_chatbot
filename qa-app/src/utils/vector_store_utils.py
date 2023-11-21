import os
from typing import List
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from utils.general_utils import read_yaml_config


class FaissConnector:
    def __init__(self, hugging_face_embedding_model_path: str, vector_store_path: str):
        self.hugging_face_embedding_model_path = hugging_face_embedding_model_path
        self.vector_store_path = vector_store_path

    @staticmethod
    def _load_document(document_file_path: str) -> List:
        return TextLoader(document_file_path).load()

    def _chunk_document(
        self, document: List, chunk_size: int, chunk_overlap: int
    ) -> List:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        return text_splitter.split_documents(document)

    def _add_user_id(self, chunked_documents: List, user_id: int) -> List:
        for doc in chunked_documents:
            doc.metadata["user_id"] = user_id
        return chunked_documents

    def _add_document_to_existing_db(
        self, chunked_documents: List, embedding_model: HuggingFaceEmbeddings
    ) -> None:
        db = FAISS.load_local(self.vector_store_path, embedding_model)
        db.add_documents(chunked_documents)
        db.save_local(self.vector_store_path)
        return None

    def _add_document_to_new_db(
        self, chunked_documents: List, embedding_model: HuggingFaceEmbeddings
    ) -> None:
        db = FAISS.from_documents(chunked_documents, embedding_model)
        db.save_local(self.vector_store_path)
        return None

    def add_document(
        self, document_file_path: str, user_id: int, chunk_size: int, chunk_overlap: int
    ):
        """Main function to add document and user id to FAISS vector store"""
        document = self._load_document(document_file_path)
        chunked_documents = self._chunk_document(document, chunk_size, chunk_overlap)
        chunked_documents_with_id = self._add_user_id(chunked_documents, user_id)
        embedding_model = HuggingFaceEmbeddings(
            model_name=self.hugging_face_embedding_model_path
        )
        if os.path.exists(self.vector_store_path + "/index.faiss"):
            self._add_document_to_existing_db(
                chunked_documents_with_id, embedding_model
            )
            print("Documents added successfully")
        else:
            self._add_document_to_new_db(chunked_documents_with_id, embedding_model)
            print("Vector store created and document added successfully")
        return None

    def load_db(self) -> FAISS:
        embedding_model = HuggingFaceEmbeddings(
            model_name=self.hugging_face_embedding_model_path
        )
        return FAISS.load_local(self.vector_store_path, embedding_model)


if __name__ == "__main__":
    yaml_config = read_yaml_config("static.yaml")
    hugging_face_embedding_model_path = yaml_config["hugging_face_embedding_model_path"]
    vector_store_path = yaml_config["vector_store_path"]
    document_file_path = yaml_config["test_document_path"]
    user_id = 1
    chunk_size = yaml_config["text_splitter"]["chunk_size"]
    chunk_overlap = yaml_config["text_splitter"]["chunk_overlap"]
    faiss_connector = FaissConnector(
        hugging_face_embedding_model_path, vector_store_path
    )
    faiss_connector.add_document(document_file_path, user_id, chunk_size, chunk_overlap)
    db = faiss_connector.load_db()
    print(f"Database chunks: {len(db.docstore._dict)}")
