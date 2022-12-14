```bash
sudo apt install tesseract-ocr libtesseract-dev -y
make env
```

or

```bash
brew install tesseract
make env
```

```bash
mkdir models
cd models
wget https://pubtables1m.blob.core.windows.net/model/pubtables1m_detection_detr_r18.pth
wget https://pubtables1m.blob.core.windows.net/model/pubtables1m_structure_detr_r18.pth
```

```bash
uvicorn src.app.api:app --host 0.0.0.0 --port 8000 --reload --reload-dir src  # dev
gunicorn src.app.api:app -c src/app/gunicorn.py -k uvicorn.workers.UvicornWorker  # prod
```

```bash
docker build -t table-detr-serving:latest -f Dockerfile .
docker run -p 8000:8000 --name table-detr-serving table-detr-serving:latest
```

```bash
curl -X 'POST' \
  'http://0.0.0.0:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@<file-path>;type=application/pdf'
```
