FROM python:3.11-slim

# NOTE: This Dockerfile does NOT install SUMO.
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV PYTHONPATH=/app
CMD ["python", "scripts/train.py", "--config", "gtqn/configs/default.yaml"]
