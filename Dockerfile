ARG PYTHON_VERSION=3.12
FROM python:$PYTHON_VERSION-slim

WORKDIR /app
COPY . /app
ENV PYTHONPATH=/app
# Install litserve and requirements
RUN pip install --no-cache-dir litserve==0.2.7 -r requirements.txt

EXPOSE 8000
CMD ["python", "/app/api/server.py"]
