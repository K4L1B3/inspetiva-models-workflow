"""Checa a deduplicacao de frames: o motor nao pode reprocessar o mesmo quadro."""

import numpy as np

from inspetiva.camera import Camera


def test_pegar_so_devolve_frame_novo():
    cam = Camera()
    assert cam.pegar() == (0, None)          # camera ainda sem frame

    cam.frame, cam.seq = np.zeros((4, 4, 3), np.uint8), 1

    seq, frame = cam.pegar(-1)
    assert seq == 1 and frame is not None
    assert frame is not cam.frame            # copia, nao a referencia viva

    assert cam.pegar(seq) == (1, None)       # mesmo quadro: nao devolve de novo

    cam.seq = 2
    seq2, frame2 = cam.pegar(seq)
    assert seq2 == 2 and frame2 is not None  # quadro novo: devolve
