import uvicorn
import uuid
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from utils.vector_store_utils import FaissConnector
from utils.general_utils import read_yaml_config

yaml_config = read_yaml_config("static.yaml")
embedding_model_path = yaml_config["hugging_face_embedding_model_path"]
vector_store_path = yaml_config["vector_store_path"]
chunk_size = yaml_config["text_splitter"]["chunk_size"]
chunk_overlap = yaml_config["text_splitter"]["chunk_overlap"]

app = FastAPI()


@app.get("/")
def main():
    content = """
    <body>
    <form action="/upload_file/" enctype="multipart/form-data" method="post">
    <input name="file" type="file">
    <input type="submit">
    </form>
    </body>
    """
    return HTMLResponse(content=content)


@app.post("/upload_file/")
async def create_test_upload_file(file: UploadFile = File(...)):
    with open(file.filename, "wb") as buffer:
        buffer.write(file.file.read())
    user_id = str(uuid.uuid4())
    try:
        faiss_connector = FaissConnector(embedding_model_path, vector_store_path)
        faiss_connector.add_document(file.filename, user_id, chunk_size, chunk_overlap)
        status = "Success"
    except Exception:
        status = "Failure"
    return {"Upload Status": status, "File Name": file.filename, "User ID": user_id}


if __name__ == "__main__":
    uvicorn.run("run_api:app", reload=True, port=8000, host="0.0.0.0")
