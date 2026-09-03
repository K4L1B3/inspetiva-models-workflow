"""Inferencia 100% local: detecta a manga, classifica a doenca.

Dois estagios:
  deteccao.pt                -> caixas de manga no frame (ultralytics)
  doenca-resnet50-v5.onnx    -> classifica cada recorte (opcional, ver MODELO_CLS)
"""

import threading
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from . import config
from .camera import aplicar_roi, camera


@lru_cache(maxsize=8)
def carregar_fonte(tamanho):
    candidatos = [config.FONT_PATH] if config.FONT_PATH else []
    candidatos += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    for c in candidatos:
        if c and Path(c).exists():
            try:
                return ImageFont.truetype(c, tamanho)
            except OSError:
                continue
    return ImageFont.load_default()


class Classificador:
    """resnet50 do Roboflow, em ONNX.

    O pre-processo e o mesmo do endpoint deles (RGB, /255, normalizacao
    ImageNet, stretch para 224x224). Conferido contra serverless.roboflow.com
    em 40 imagens do split de teste: 40/40 identicos.
    """

    MEDIA = np.array([0.485, 0.456, 0.406], np.float32)
    DESVIO = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, caminho):
        self.sessao = ort.InferenceSession(str(caminho))
        self.entrada = self.sessao.get_inputs()[0].name
        self.saida = self.sessao.get_outputs()[0].name

    def __call__(self, recorte_bgr):
        x = cv2.resize(recorte_bgr, (224, 224))[:, :, ::-1].astype(np.float32)
        x = (x / 255.0 - self.MEDIA) / self.DESVIO
        entrada = np.ascontiguousarray(x.transpose(2, 0, 1)[None])

        # ponytail: o ONNX veio com batch fixo em 1, entao e um recorte por vez.
        # Sao poucas mangas por frame; se virar gargalo, reexportar com batch
        # dinamico ou trocar para onnxruntime-gpu.
        logits = self.sessao.run([self.saida], {self.entrada: entrada})[0][0]
        exp = np.exp(logits - logits.max())
        prob = exp / exp.sum()
        i = int(prob.argmax())
        return config.CLASSES_DOENCA[i], float(prob[i])


class Modelo:
    """Carrega os pesos uma vez e infere sobre um frame BGR."""

    def __init__(self):
        self.det = YOLO(str(config.MODELO_DET))
        self.cls = Classificador(config.MODELO_CLS) if config.MODELO_CLS else None
        self.device = config.DEVICE or None

    def inferir(self, frame_bgr):
        """Devolve deteccoes no formato {class, confidence, x, y, width, height}."""
        r = self.det.predict(frame_bgr, imgsz=config.IMGSZ, conf=config.CONF_DET,
                             device=self.device, verbose=False)[0]

        caixas, recortes = [], []
        h, w = frame_bgr.shape[:2]
        for b in r.boxes:
            x0, y0, x1, y1 = (int(v) for v in b.xyxy[0].tolist())
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            caixas.append({
                "class": r.names[int(b.cls)],
                "confidence": float(b.conf),
                "x": (x0 + x1) / 2, "y": (y0 + y1) / 2,
                "width": x1 - x0, "height": y1 - y0,
            })
            recortes.append(frame_bgr[y0:y1, x0:x1])

        if self.cls:
            for caixa, recorte in zip(caixas, recortes):
                caixa["class"], caixa["confidence"] = self.cls(recorte)

        return caixas


def separar(deteccoes):
    """Divide pelo limiar: acima e classificado, abaixo e inconclusivo."""
    ok = [d for d in deteccoes if d["confidence"] >= config.LIMIAR]
    inc = [d for d in deteccoes if d["confidence"] < config.LIMIAR]
    return ok, inc


def contar(classificadas):
    contagem = {}
    for det in classificadas:
        nome = config.NOMES.get(det["class"], det["class"])
        contagem[nome] = contagem.get(nome, 0) + 1
    return contagem


def anotar(frame_bgr, classificadas, inconclusivas):
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    escala = max(img.width / 1280, 0.8)
    fonte = carregar_fonte(int(22 * escala))
    grossura = max(int(3 * escala), 2)

    for deteccoes, cor, sufixo in (
        (classificadas, config.AMBAR, None),
        (inconclusivas, config.CINZA, "inconclusivo"),
    ):
        for det in deteccoes:
            cx, cy, w, h = det["x"], det["y"], det["width"], det["height"]
            x0, y0 = cx - w / 2, cy - h / 2
            d.rectangle([x0, y0, cx + w / 2, cy + h / 2], outline=cor,
                        width=grossura)

            nome = sufixo or config.NOMES.get(det["class"], det["class"])
            texto = f"{nome}  {det['confidence']:.0%}"

            cx0, cy0, cx1, cy1 = d.textbbox((0, 0), texto, font=fonte)
            tw, th = cx1 - cx0, cy1 - cy0
            pad = int(6 * escala)
            ty = max(0, y0 - th - pad * 2)
            d.rectangle([x0, ty, x0 + tw + pad * 2, ty + th + pad * 2], fill=cor)
            d.text((x0 + pad, ty + pad), texto, font=fonte, fill=config.GRAFITE)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


class Motor:
    """Roda a inferencia num thread proprio sobre o ultimo frame da camera.

    O stream de video nunca espera o modelo: ele so le o ultimo quadro pronto.
    """

    def __init__(self):
        self.modelo = None
        self.jpeg = None
        self.stats = {"detectados": 0, "classificados": 0, "inconclusivos": 0,
                      "contagem": {}, "ms": 0, "pronto": False}
        self.erro = None
        self.lock = threading.Lock()
        self.thread = None

    def iniciar(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
        return self

    def _loop(self):
        try:
            self.modelo = Modelo()
        except Exception as exc:              # peso faltando, CUDA quebrada...
            self.erro = f"falha ao carregar os pesos: {exc}"
            return

        seq = -1
        while True:
            seq, frame = camera.pegar(seq)
            if frame is None:      # nada novo: nao reprocessa o mesmo quadro
                time.sleep(0.01)
                continue

            inicio = time.time()
            recorte = aplicar_roi(frame)
            try:
                deteccoes = self.modelo.inferir(recorte)
                self.erro = None
            except Exception as exc:
                self.erro = f"falha na inferencia: {exc}"
                time.sleep(1.0)
                continue

            ok, inc = separar(deteccoes)
            anotado = anotar(recorte, ok, inc)
            if anotado.shape[1] > 1280:
                k = 1280 / anotado.shape[1]
                anotado = cv2.resize(anotado, None, fx=k, fy=k)
            feito, buf = cv2.imencode(".jpg", anotado,
                                      [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not feito:
                continue

            with self.lock:
                self.jpeg = buf.tobytes()
                self.stats = {
                    "detectados": len(deteccoes),
                    "classificados": len(ok),
                    "inconclusivos": len(inc),
                    "contagem": contar(ok),
                    "ms": int((time.time() - inicio) * 1000),
                    "pronto": True,
                }

    def quadro(self):
        with self.lock:
            return self.jpeg

    def instantaneo(self):
        with self.lock:
            return dict(self.stats, erro=self.erro)


def stream():
    """MJPEG do que o modelo esta enxergando agora."""
    ultimo = None
    while True:
        jpeg = motor.quadro()
        if jpeg is None or jpeg is ultimo:   # so manda quadro novo
            time.sleep(0.02)
            continue
        ultimo = jpeg
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")


motor = Motor()
