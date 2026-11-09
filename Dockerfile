FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SGL_DATA_DIR=/data

WORKDIR /srv/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sgl ./sgl
COPY app ./app

# Данные набора Higgs в образ не входят: каталог подключается при запуске (-v ./data:/data).
VOLUME ["/data"]
EXPOSE 8000

# База данных и авторизация не используются.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
