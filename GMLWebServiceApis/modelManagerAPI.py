import sys
import os
kgnet_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
inference_dir = os.path.abspath(os.path.dirname(__file__))

sys.path.append(kgnet_dir)
sys.path.append(inference_dir)

from Constants import *
from uvicorn import run
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse

app = FastAPI()
HOST = '127.0.0.1'
# HOST = '0.0.0.0'
PORT = KGNET_Config.GML_ModelManager_PORT

@app.post("/uploadmodel/")
async def saveIncomingModel(file: UploadFile = File(...)):
    filepath = os.path.join(Constants.KGNET_Config.trained_model_path,
                            file.filename)
    with open(filepath, "wb") as f:
        f.write(file.file.read())
    return JSONResponse(content={"message": "Model uploaded successfully"})


@app.get("/downloadmodel/{mid}")
async def sendTrainedModel(mid:str):
    filepath = os.path.join(Constants.KGNET_Config.trained_model_path,mid)
    if not os.path.exists(filepath):
        return {"error": "File not found"}
    return FileResponse(filepath, headers={"Content-Disposition": f"attachment; filename={mid}"})

if __name__ == "__main__":
    run(app,host=HOST,port=PORT)
