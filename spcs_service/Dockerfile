FROM python:3.11-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    huggingface_hub==0.21.4 \
    torch==2.1.0 \
    torchaudio==2.1.0 \
    speechbrain==1.0.0 \
    snowflake-connector-python[pandas]==3.6.0 \
    snowflake-snowpark-python==1.11.1 \
    flask==3.0.0 \
    gunicorn==21.2.0 \
    numpy==1.24.3 \
    scipy \
    pydub==0.25.1 \
    requests==2.31.0

# Pre-download speechbrain model (NOT gated)
RUN python3 -c "\
from speechbrain.inference.speaker import EncoderClassifier; \
print('Downloading speechbrain ECAPA-TDNN model...'); \
classifier = EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir='/root/.cache/speechbrain/spkrec-ecapa-voxceleb'); \
print('Model downloaded successfully'); \
"

# Create app directory
WORKDIR /app

# Copy application files
COPY speaker_service.py /app/speaker_service.py
COPY config.py /app/

# Expose port for the service
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the service
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "300", "speaker_service:app"]
