FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/app
COPY config/ /app/config
COPY models/ /app/models

ENV PYTHONPATH=/app
ENV APP_CONFIG=/app/config/app.yaml
ENV POLICY_CONFIG=/app/config/policy.yaml

CMD ["python", "-m", "app.main"]