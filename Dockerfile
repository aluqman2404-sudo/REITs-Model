FROM python:3.12.9-slim

# Monitoring configuration
# ENV LOG_DESTINATION=stdout          # stdout | cloudwatch | file | datadog
# ENV CLOUDWATCH_LOG_GROUP=/uk-housing-model/app  # required if LOG_DESTINATION=cloudwatch
# ENV ENV=production                  # dev | staging | production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/mpl \
    LOG_LEVEL=INFO

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY . /app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
