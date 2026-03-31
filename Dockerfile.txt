# -------------------------------------------------------------------
# 🧠 Fallacy Analyzer - Production Dockerfile
# Maintained for Walter E. Ward | 2025
# -------------------------------------------------------------------

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app.py /app/app.py

ENV APP_SECRET_KEY="super_secure_fallback_key"
ENV ADMIN_EMAIL="your_email@gmail.com"
ENV ALERT_RECIPIENT="your_email@gmail.com"
ENV EMAIL_PASSWORD="your_app_password_here"
ENV HF_API_TOKEN="your_huggingface_token_here"
ENV TRANSFORMERS_CACHE="/app/.cache"

RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    transformers==4.44.0 \
    flask waitress \
    cryptography \
    huggingface_hub \
    smtplib email-validator \
    && python -m nltk.downloader punkt

ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 5000

CMD ["python", "app.py"]
