# Base image
FROM continuumio/miniconda3

# Install dependencies
WORKDIR /mlops
COPY environment.yml environment.yml
COPY src src
COPY setup.py setup.py
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc build-essential tesseract-ocr libtesseract-dev \
    && conda update conda -y -q \
    && conda env create -f environment.yml -q

# Copy
COPY models models

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "table-detr", "/bin/bash", "-c"]

# Export ports
EXPOSE 8000

# Start app
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "table-detr", "gunicorn", "src.app.api:app", "-c", "src/app/gunicorn.py", "-k", "uvicorn.workers.UvicornWorker"]