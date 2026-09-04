# Jenkins는 같은 파일의 별도 빌드 단계이며 애플리케이션 이미지와 섞이지 않는다.
FROM docker:cli AS dockercli
FROM jenkins/jenkins:lts-jdk21 AS jenkins
USER root
COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=dockercli /usr/local/libexec/docker/cli-plugins/docker-compose /usr/local/libexec/docker/cli-plugins/docker-compose
USER jenkins
RUN jenkins-plugin-cli --plugins workflow-aggregator git github credentials-binding
COPY jenkins-init.groovy /usr/share/jenkins/ref/init.groovy.d/hydrotwin.groovy

# 기본 빌드 대상은 Python 애플리케이션이다.
FROM python:3.11-slim AS app
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY . /app
CMD ["python", "-m", "src.runtime.server_entry", "api"]
