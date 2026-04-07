FROM python:3.11-slim@sha256:1d9786cbecd37244b212ad969b34b49065b15c0966a2a044a4d04bc071f8924b

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860 8000

CMD ["python", "space_app.py"]