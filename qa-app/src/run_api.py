import uvicorn
from fastapi import FastAPI, File, UploadFile
from langchain.document_loaders import TextLoader
from fastapi.responses import HTMLResponse

app = FastAPI()


@app.post("/test_upload_file/")
async def create_test_upload_file(file: UploadFile = File(...)):
    with open(file.filename, "wb") as buffer:
        buffer.write(file.file.read())
    loader = TextLoader(file.filename)
    document = loader.load()
    return {"File loaded": document[0].metadata}


@app.post("/upload_file/")
async def create_upload_file(file: UploadFile = File(...)):
    with open(file.filename, "wb") as buffer:
        buffer.write(file.file.read())
    loader = TextLoader(file.filename)
    document = loader.load()

    return {"content": document[0].metadata}


@app.get("/")
def main():
    content = """
    <body>
    <form action="/uploadfile/" enctype="multipart/form-data" method="post">
    <input name="file" type="file">
    <input type="submit">
    </form>
    </body>
    """
    return HTMLResponse(content=content)


if __name__ == "__main__":
    uvicorn.run("run_api:app", reload=True, port=8000, host="0.0.0.0")
