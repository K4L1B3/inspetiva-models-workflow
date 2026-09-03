# Inspetiva IS-01

Estação de demonstração (Agrinordeste): câmera USB + **inferência 100% local**.
Nada sai da máquina — sem API key, sem internet, sem servidor de inferência.

| Rota | Para quê |
|---|---|
| `/` | Página única: vídeo já anotado pelo modelo. |
| `/video` | Só o stream MJPEG anotado (embutir na TV). |
| `/video-cru` | Câmera sem inferência, com o retângulo do ROI, para alinhar a bandeja. |
| `/status` | JSON: câmera, device, contagens, ms por frame, erro do modelo. |

## Rodar

Requer [uv](https://docs.astral.sh/uv/) e uma webcam USB.

```bash
uv sync                 # cria .venv e instala tudo
cp .env.example .env    # ajuste a câmera e o ROI
uv run inspetiva        # sobe em http://0.0.0.0:8000
```

Testes: `uv run pytest -q`

## Modelos

> Os pesos **não estão no repositório** (`models/` é ignorado pelo git — 95 MB).
> Copie `deteccao.pt` e `doenca-resnet50-v5.onnx` para `models/` antes de rodar.

Dois estágios, ambos em `models/`, carregados pelo ultralytics:

| Arquivo | O que é | Treino | Métrica |
|---|---|---|---|
| `deteccao.pt` | detector de manga (yolo26n, 640px, ultralytics) | dataset `TCC - Detector de Manga` v5, 100 épocas | mAP50 **0,947** · P 0,954 · R 0,917 |
| `doenca-resnet50-v5.onnx` | classificador de doença (resnet50, 224px, onnxruntime) | treinado no Roboflow, dataset `TCC - Doença em Fruto` v5 | top-1 **96,1%** (valid) · 92,5% (test) |

O detector devolve as caixas; cada recorte passa pelo classificador. Abaixo de
`LIMIAR_CLASSIFICACAO` a caixa vira "inconclusivo" (cinza).

O classificador é o modelo do Roboflow rodando localmente: o `.onnx` foi
baixado uma vez e `inspetiva/modelo.py` reproduz o pré-processo deles (RGB,
`/255`, normalização ImageNet, stretch para 224×224). Conferido contra
`serverless.roboflow.com` em 40 imagens do split de teste — **40/40 idênticos**.
A ordem de `CLASSES_DOENCA` no `config.py` é o `CLASS_MAP` do export e não pode
ser reordenada: o ONNX não carrega os nomes dentro dele.

Os pesos do detector **não são baixáveis** do Roboflow — lá ele está como
"yolov8n Model Upload", ou seja, foi treinado fora e subido; o `deteccao.pt`
daqui é o original.

### Desempenho (GTX local, `DEVICE=0`)

Pipeline completo com uma manga no frame: **~35 ms** (≈28 fps). O detector roda
na GPU, mas o `onnxruntime` instalado é o de CPU e o classificador custa ~25 ms
por recorte — com 4 mangas o frame vai a ~110 ms (≈9 fps). Se precisar de mais,
troque por `onnxruntime-gpu`.

## Configuração (`.env`)

| Variável | Padrão | Nota |
|---|---|---|
| `CAM_DEVICE` / `CAM_INDEX` / `CAM_W` / `CAM_H` / `CAM_FPS` | vazio / `0` / `1280` / `720` / `30` | No Linux, `CAM_DEVICE` aceita o caminho persistente de `/dev/v4l/by-id` e tem prioridade sobre o índice. |
| `FOCO_FIXO` / `FOCO` | `1` / `30` | Autofoco caça durante a demo e borra o fruto. |
| `ROI_X/Y/W/H` | `0` | Recorte fixo da bandeja. Zeros = frame inteiro. Alinhe por `/video-cru`. |
| `MODELO_DET` / `MODELO_CLS` | `models/deteccao.pt` / `models/doenca-resnet50-v5.onnx` | `MODELO_CLS` vazio = só o detector, sem o estágio de doença. |
| `DEVICE` | vazio | Vazio seleciona automaticamente; `0` força a primeira GPU NVIDIA; `cpu` força CPU. |
| `IMGSZ` / `CONF_DET` | `640` / `0.40` | Entrada e confiança mínima do detector. |
| `LIMIAR_CLASSIFICACAO` | `0.70` | Abaixo disso a detecção vira "inconclusivo" (caixa cinza). |
| `FONT_PATH`, `HOST`, `PORT` | — / `0.0.0.0` / `8000` | Fonte cai em DejaVu/Arial se vazio. |

## Estrutura

```
inspetiva/
  config.py    env, paleta, nomes das classes
  camera.py    thread de captura, ROI, stream MJPEG
  modelo.py    dois estágios ultralytics, limiar, anotação, thread de inferência
  main.py      rotas FastAPI (entry point `inspetiva`)
  static/      index.html
models/        pesos locais
fixtures/      uma imagem por classe, do split de teste — usadas no pytest
test_inspetiva.py
```

## Checagem antes da feira

1. `curl localhost:8000/status` → `camera: true`, `pronto: true`, `erro: null`.
2. Abra `/video-cru`, ajuste `ROI_*` no `.env`, reinicie.
3. Confira `ms` no `/status`: com `DEVICE=0` na GPU deve ficar bem abaixo de 100 ms.
