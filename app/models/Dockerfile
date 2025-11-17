FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y openjdk-17-jdk-headless wget gnupg2 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Thiết lập biến môi trường JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV JVM_PATH="${JAVA_HOME}/lib/server/libjvm.so"
ENV LD_LIBRARY_PATH="${JAVA_HOME}/lib/server"
ENV PYJNIUS_JVM="${JAVA_HOME}/lib/server/libjvm.so"

WORKDIR /app
COPY requirements.txt /app/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app
EXPOSE 8000


CMD ["python", "app.py"]
