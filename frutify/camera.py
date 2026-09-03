"""Captura da webcam num thread proprio + stream MJPEG para o preview."""

import os
import threading
import time

import cv2

from . import config


class Camera:
    """Le a C920 num thread proprio e guarda sempre o ultimo frame."""

    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.ok = False
        self.cap = None
        self.thread = None

    def iniciar(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
        return self

    def _abrir(self):
        # CAP_DSHOW e obrigatorio no Windows. Sem MJPG a C920 entrega
        # 5 fps em 1080p, porque cai em YUY2 nao comprimido.
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
        cap = cv2.VideoCapture(config.CAM_SOURCE, backend)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_H)
        cap.set(cv2.CAP_PROP_FPS, config.CAM_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Foco fixo: autofoco caca durante a demo e borra o fruto.
        if config.FOCO_FIXO:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            cap.set(cv2.CAP_PROP_FOCUS, config.FOCO)
        return cap

    def _loop(self):
        while True:
            if self.cap is None or not self.cap.isOpened():
                self.cap = self._abrir()
                time.sleep(0.5)
            ret, frame = self.cap.read()
            if not ret:
                self.ok = False
                self.cap.release()
                self.cap = None
                time.sleep(1.0)
                continue
            with self.lock:
                self.frame = frame
                self.ok = True

    def pegar(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


camera = Camera()


def aplicar_roi(frame):
    x, y, w, h = config.ROI
    if w <= 0 or h <= 0:
        return frame
    fh, fw = frame.shape[:2]
    x, y = max(0, x), max(0, y)
    w, h = min(w, fw - x), min(h, fh - y)
    return frame[y:y + h, x:x + w]


def gerar_mjpeg(marcar_roi: bool):
    while True:
        frame = camera.pegar()
        if frame is None:
            time.sleep(0.05)
            continue
        if marcar_roi and config.ROI[2] > 0:
            x, y, w, h = config.ROI
            cv2.rectangle(frame, (x, y), (x + w, y + h), config.AMBAR[::-1], 2)
        else:
            frame = aplicar_roi(frame)
        alvo = 1280
        if frame.shape[1] > alvo:
            k = alvo / frame.shape[1]
            frame = cv2.resize(frame, None, fx=k, fy=k)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
        time.sleep(1 / 25)
