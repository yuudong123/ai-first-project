#!/bin/bash

PROJECT_DIR="/home/project/ai-first-project-kafka-test"
PYTHON="$PROJECT_DIR/.venv/bin/python"
CONSUMER="$PROJECT_DIR/kafka/raw_consumer.py"
LOG="$PROJECT_DIR/kafka/logs/raw_consumer.log"

mkdir -p "$PROJECT_DIR/kafka/logs"

while true
do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Raw Consumer START" >> "$LOG"

    "$PYTHON" -u "$CONSUMER" >> "$LOG" 2>&1

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Raw Consumer STOP - retry in 5 sec" >> "$LOG"

    sleep 5
done
