```bash
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
make env
```

or

```bash
brew install tesseract
make env
```

```bash
uvicorn src.app.api:app --host 0.0.0.0 --port 8000 --log-config src/app/log.ini --reload --reload-dir src  # dev
gunicorn src.app.api:app -c src/app/gunicorn.py -k uvicorn.workers.UvicornWorker --log-config src/app/log.ini  # prod
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
    -F 'file=@<file-path>;type=image/jpeg'
```
