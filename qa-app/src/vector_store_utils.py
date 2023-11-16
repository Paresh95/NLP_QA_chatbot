import os
from typing import List
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from utils import read_yaml_config


def chunk_documents(document_file_path: str, id: int, yaml_config: dict) -> List:
    loader = TextLoader(document_file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=yaml_config["text_splitter"]["chunk_size"],
        chunk_overlap=yaml_config["text_splitter"]["chunk_overlap"],
        length_function=len,
        add_start_index=True,
    )
    docs = text_splitter.split_documents(documents)
    for doc in docs:
        doc.metadata["user_id"] = id
    return docs


def upload_document_to_faiss_db(
    document_file_path: str, id: int, yaml_config: dict
) -> None:
    docs = chunk_documents(document_file_path, id, yaml_config)
    embedding_model = HuggingFaceEmbeddings(
        model_name=yaml_config["hugging_face_embedding_model_path"]
    )

    if os.path.exists(yaml_config["vector_store_path"] + "/index.faiss"):
        db = FAISS.load_local(yaml_config["vector_store_path"], embedding_model)
        db.add_documents(docs)
        db.save_local(yaml_config["vector_store_path"])
        print("Documents added successfully")
    else:
        db = FAISS.from_documents(docs, embedding_model)
        db.save_local(yaml_config["vector_store_path"])
        print("Vector store created and document added successfully")
    return None


def load_faiss_db(yaml_config: dict) -> FAISS:
    embedding_model = HuggingFaceEmbeddings(
        model_name=yaml_config["hugging_face_embedding_model_path"]
    )
    db = FAISS.load_local(yaml_config["vector_store_path"], embedding_model)
    return db


if __name__ == "__main__":
    yaml_config = read_yaml_config("static.yaml")
    upload_document_to_faiss_db("data/Transcript Otter - A1.txt", 1, yaml_config)
    db = load_faiss_db(yaml_config)
    print(len(db.docstore._dict))
