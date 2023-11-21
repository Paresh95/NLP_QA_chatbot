import uvicorn
import uuid
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from src.utils.vector_store_utils import FaissConnector
from src.utils.general_utils import read_yaml_config
from src.utils.rag_utils import RagSystem

yaml_config = read_yaml_config("parameters.yaml")
hugging_face_embedding_model_path = yaml_config["hugging_face_embedding_model_path"]
vector_store_path = yaml_config["vector_store_path"]
chunk_size = yaml_config["text_splitter"]["chunk_size"]
chunk_overlap = yaml_config["text_splitter"]["chunk_overlap"]

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def main():
    return FileResponse("templates/index.html")


@app.post("/upload_file/")
async def upload_file_to_db(file: UploadFile = File(...)):
    with open(file.filename, "wb") as buffer:
        buffer.write(file.file.read())
    user_id = str(uuid.uuid4())
    try:
        faiss_connector = FaissConnector(
            hugging_face_embedding_model_path, vector_store_path
        )
        faiss_connector.add_document(file.filename, user_id, chunk_size, chunk_overlap)
        status = "Success"
    except Exception as e:
        status = f"Failure: {e}"
    return {"Upload Status": status, "File Name": file.filename, "User ID": user_id}


@app.post("/query_file/")
async def query_file(query: str = Form(...), user_id: str = Form(...)):
    try:
        results = RagSystem(config=yaml_config, user_id=user_id).run_query(query)
        data = {
            "User ID": user_id,
            "Question": results["question"],
            "Answer": results["answer"],
            # "Source Documents": results["source_documents"]
        }
        return data
    except Exception as e:
        return {"Error": str(e)}


@app.get("/retrieve_docs/")
async def retrieve_documents(query: str, user_id: str):
    try:
        faiss_connector = FaissConnector(
            hugging_face_embedding_model_path, vector_store_path
        )
        db = faiss_connector.load_db()
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"filter": {"user_id": user_id}, "k": 5},
        )
        retrieved_docs = retriever.invoke(query)
        data = {
            "User ID": user_id,
            "Query": query,
            "Retrieved Documents": retrieved_docs,
            "No: chunks in db": len(db.docstore._dict),
        }
        return data
    except Exception as e:
        return {"Error": str(e)}


if __name__ == "__main__":
    uvicorn.run("run_api:app", reload=True, port=8000, host="0.0.0.0")
