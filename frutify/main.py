"""
Frutify IS-01 — inferencia local, ao vivo.

Abra http://localhost:8000 : video em tempo real com as caixas do modelo.
Nada sai da maquina — os pesos estao em models/.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from . import config, modelo
from .camera import camera, gerar_mjpeg


@asynccontextmanager
async def lifespan(_app):
    camera.iniciar()      # thread de captura
    modelo.motor.iniciar()  # thread de inferencia (carrega os pesos)
    yield


app = FastAPI(title="Frutify IS-01", lifespan=lifespan)

MJPEG = "multipart/x-mixed-replace; boundary=frame"


@app.get("/")
def index():
    return FileResponse(config.STATIC / "index.html")


@app.get("/video")
def video():
    """O que o modelo esta enxergando: frame anotado, em tempo real."""
    return StreamingResponse(modelo.stream(), media_type=MJPEG)


@app.get("/video-cru")
def video_cru():
    """Camera sem inferencia, com o retangulo do ROI, para alinhar a bandeja."""
    return StreamingResponse(gerar_mjpeg(marcar_roi=True), media_type=MJPEG)


@app.get("/status")
def status():
    return {
        "camera": camera.ok,
        "camera_source": str(config.CAM_SOURCE),
        "roi": config.ROI,
        "limiar": config.LIMIAR,
        "device": config.DEVICE or "auto",
        **modelo.motor.instantaneo(),
    }


def run():
    """Entry point: `uv run frutify`."""
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="warning")


if __name__ == "__main__":
    run()
