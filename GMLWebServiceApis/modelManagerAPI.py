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
from fastapi import Response,status

app = FastAPI()
HOST = '127.0.0.1'
# HOST = '0.0.0.0'
PORT = KGNET_Config.GML_ModelManager_PORT

@app.post("/uploadmodel/")
async def saveIncomingModel(model: UploadFile = File(...)):
    filepath = os.path.join(KGNET_Config.trained_model_path,
                            model.filename)
    with open(filepath, "wb") as f:
        f.write(model.file.read())
    print(f'model {model.filename} saved successfully')
    return JSONResponse(content={"message": "Model uploaded successfully"})

@app.post("/uploaddataset/")
async def saveIncomingDataset(dataset: UploadFile = File(...)):
    filepath = os.path.join(KGNET_Config.datasets_output_path,
                            dataset.filename)
    with open(filepath, "wb") as f:
        f.write(dataset.file.read())
    print(f'Dataset {dataset.filename} saved successfully')
    return JSONResponse(content={"message": "Model uploaded successfully"})


@app.get("/downloadmodel/{mid}")
async def sendTrainedModel(mid:str,response:Response):
    filepath = os.path.join(KGNET_Config.trained_model_path,mid)
    if not os.path.exists(filepath):
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": "File not found"}
    return FileResponse(filepath, headers={"Content-Disposition": f"attachment; filename={mid}"})

@app.get("/downloaddataset/{mid}")
async def sendDataset(mid:str,response:Response):
    # mid = 'mid-' + utils.getIdWithPaddingZeros(mid)
    filepath = os.path.join(KGNET_Config.datasets_output_path,mid)
    if not os.path.exists(filepath):
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": "File not found"}
    return FileResponse(filepath, headers={"Content-Disposition": f"attachment; filename={mid}"})

if __name__ == "__main__":
    run(app,host=HOST,port=PORT)
