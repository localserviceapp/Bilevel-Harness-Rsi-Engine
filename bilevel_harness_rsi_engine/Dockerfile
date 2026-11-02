FROM python:3.11-slim

WORKDIR /app

# Create non-root sandbox user for Gate 3 isolation
RUN groupadd -r sandbox && useradd -r -g sandbox -d /home/sandbox -s /sbin/nologin sandbox \
    && mkdir -p /home/sandbox && chown -R sandbox:sandbox /home/sandbox

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
