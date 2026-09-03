"""Check minimo: a logica que quebra silenciosamente durante a demo."""

import cv2
import numpy as np
import pytest

from inspetiva import config, modelo


def det(cls, conf, x=100, y=100, w=60, h=60):
    return {"class": cls, "confidence": conf, "x": x, "y": y,
            "width": w, "height": h}


def test_separar_usa_o_limiar():
    ok, inc = modelo.separar([det("Healthy", 0.9), det("Scab", 0.4)])
    assert [d["class"] for d in ok] == ["Healthy"]
    assert [d["class"] for d in inc] == ["Scab"]


def test_contar_traduz_nomes():
    assert modelo.contar([det("Healthy", 1), det("Healthy", 1)]) == {"Sadio": 2}
    assert modelo.contar([det("Bicho Novo", 1)]) == {"Bicho Novo": 1}


def test_anotar_devolve_frame_do_mesmo_tamanho():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    saida = modelo.anotar(frame, [det("Scab", 0.9)], [det("Healthy", 0.1)])
    assert saida.shape == frame.shape and saida.any()   # desenhou algo


def test_pesos_existem():
    assert config.MODELO_DET.exists()
    assert config.MODELO_CLS is None or config.MODELO_CLS.exists()


def test_modelo_infere_num_frame_sintetico():
    """Carrega os pesos de verdade e roda os dois estagios uma vez."""
    m = modelo.Modelo()
    saida = m.inferir(np.full((480, 640, 3), 120, dtype=np.uint8))
    assert isinstance(saida, list)
    for d in saida:
        assert {"class", "confidence", "x", "y", "width", "height"} <= d.keys()


# Uma imagem real por classe, do split de teste do dataset v5. Se alguem
# mexer no pre-processo (RGB/BGR, /255, normalizacao ImageNet) a acuracia
# despenca para ~50-69% e pelo menos uma destas para de bater.
@pytest.mark.parametrize("esperado", config.CLASSES_DOENCA)
def test_classificador_acerta_uma_imagem_de_cada_classe(esperado):
    if config.MODELO_CLS is None:
        pytest.skip("estagio de doenca desligado")
    img = cv2.imread(str(config.RAIZ / "fixtures" /
                         f"{esperado.replace(' ', '_')}.jpg"))
    assert img is not None
    classe, conf = modelo.Classificador(config.MODELO_CLS)(img)
    assert classe == esperado and conf > 0.5


if __name__ == "__main__":
    import sys

    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
