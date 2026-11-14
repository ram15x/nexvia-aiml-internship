FROM python:3.12-slim
WORKDIR /app
# TensorFlow, FAISS and the API only; Spark/Airflow are offline tools and are not needed to serve
RUN pip install --no-cache-dir numpy scipy pandas pyarrow "tensorflow-cpu==2.21.0" "tf-keras==2.21.0" "tensorflow-recommenders==0.7.7" \
    "faiss-cpu==1.15.1" redis fastapi uvicorn httpx
COPY src /app/src
ENV PYTHONPATH=/app/src TF_USE_LEGACY_KERAS=1 TF_CPP_MIN_LOG_LEVEL=2 BUNDLE=/bundle
EXPOSE 8000
CMD ["uvicorn", "rec.api:app", "--host", "0.0.0.0", "--port", "8000"]
