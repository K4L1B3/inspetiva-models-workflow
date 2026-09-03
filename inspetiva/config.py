"""Configuracao lida do .env. Um unico lugar para tunar a estacao."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).parent
STATIC = BASE / "static"
RAIZ = BASE.parent

# ------------------------------------------------------------------ camera

CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
# No Linux, prefira o link persistente de /dev/v4l/by-id para que a camera
# selecionada nao mude quando outro dispositivo USB for conectado.
CAM_DEVICE = os.getenv("CAM_DEVICE", "").strip()
CAM_SOURCE = CAM_DEVICE or CAM_INDEX
CAM_W = int(os.getenv("CAM_W", "1280"))
CAM_H = int(os.getenv("CAM_H", "720"))
CAM_FPS = int(os.getenv("CAM_FPS", "30"))
FOCO_FIXO = os.getenv("FOCO_FIXO", "1") == "1"
FOCO = float(os.getenv("FOCO", "30"))

# Region of Interest: recorte fixo da bandeja. Zeros = usa o frame inteiro.
ROI = (
    int(os.getenv("ROI_X", "0")),
    int(os.getenv("ROI_Y", "0")),
    int(os.getenv("ROI_W", "0")),
    int(os.getenv("ROI_H", "0")),
)

# ------------------------------------------------------------------ modelos

# Pesos locais (YOLO/ultralytics). Nada sai da maquina.
MODELO_DET = RAIZ / os.getenv("MODELO_DET", "models/deteccao.pt")
# Classificador de doenca em ONNX (resnet50 treinado no Roboflow).
# Vazio = so o detector, sem o segundo estagio.
_cls = os.getenv("MODELO_CLS", "models/doenca-resnet50-v5.onnx")
MODELO_CLS = (RAIZ / _cls) if _cls else None

# A ordem importa: e o CLASS_MAP com que o modelo foi exportado, e o ONNX
# nao carrega os nomes dentro dele.
CLASSES_DOENCA = ["Healthy", "Stem End Rot", "Antracnose-Ceratitis",
                  "Bacterial Canker", "Scab"]

DEVICE = os.getenv("DEVICE", "")          # "" = auto (cuda se houver), "cpu", "0"
IMGSZ = int(os.getenv("IMGSZ", "640"))
CONF_DET = float(os.getenv("CONF_DET", "0.40"))   # confianca minima do detector
LIMIAR = float(os.getenv("LIMIAR_CLASSIFICACAO", "0.70"))  # abaixo = inconclusivo

# ------------------------------------------------------------------- visual

FONT_PATH = os.getenv("FONT_PATH", "")

GRAFITE = (44, 44, 42)
AMBAR = (239, 159, 39)
OSSO = (241, 239, 232)
CINZA = (138, 136, 130)

NOMES = {
    # classificador de doenca
    "Antracnose-Ceratitis": "Antracnose",
    "Bacterial Canker": "Cancro bacteriano",
    "Stem End Rot": "Podridao peduncular",
    "Scab": "Sarna",
    "Healthy": "Sadio",
    # detector
    "Fresh Mango": "Manga sadia",
    "Rotten Mango": "Manga podre",
    "Mango": "Manga",
}

# ------------------------------------------------------------------ servidor

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
