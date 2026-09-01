#!/usr/bin/env bash
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
PRODUCER_SCRIPT="${PROJECT_ROOT}/kafka/virtual_factory_producer_v5.py"
CONSUMER_SCRIPT="${PROJECT_ROOT}/kafka/raw_consumer.py"
RUN_DIR="${PROJECT_ROOT}/kafka/run"
LOG_DIR="${PROJECT_ROOT}/kafka/logs"
PRODUCER_PID_FILE="${RUN_DIR}/v5_normal_producer.pid"
CONSUMER_PID_FILE="${RUN_DIR}/raw_consumer.pid"
PRODUCER_LOG="${LOG_DIR}/v5_normal_producer.log"
CONSUMER_LOG="${LOG_DIR}/raw_consumer.log"

process_matches() {
    local pid="$1"
    local marker="$2"
    [ -r "/proc/${pid}/cmdline" ] || return 1
    tr '\0' ' ' < "/proc/${pid}/cmdline" | grep -Fq -- "${marker}"
}

component_status() {
    local name="$1"
    local pid_file="$2"
    local marker="$3"
    if [ -f "${pid_file}" ]; then
        local pid
        pid="$(tr -d '[:space:]' < "${pid_file}")"
        if [ -n "${pid}" ] && process_matches "${pid}" "${marker}"; then
            echo "${name}: RUNNING pid=${pid}"
            return 0
        fi
    fi
    echo "${name}: STOPPED"
    return 1
}

start_component() {
    local name="$1"
    local pid_file="$2"
    local marker="$3"
    local log_file="$4"
    local script="$5"
    shift 5

    if component_status "${name}" "${pid_file}" "${marker}" >/dev/null; then
        component_status "${name}" "${pid_file}" "${marker}"
        return 0
    fi

    local existing
    existing="$(pgrep -f -- "${marker}" | head -n 1 || true)"
    if [ -n "${existing}" ]; then
        echo "${name}: unmanaged matching process exists pid=${existing}; refusing duplicate start"
        return 1
    fi

    rm -f -- "${pid_file}"
    nohup setsid "${PYTHON_BIN}" -u "${script}" "$@" >> "${log_file}" 2>&1 < /dev/null &
    local pid=$!
    echo "${pid}" > "${pid_file}"
    sleep 3
    if process_matches "${pid}" "${marker}"; then
        echo "${name}: STARTED pid=${pid}"
        return 0
    fi

    echo "${name}: FAILED TO START; check ${log_file}"
    rm -f -- "${pid_file}"
    return 1
}

stop_component() {
    local name="$1"
    local pid_file="$2"
    local marker="$3"
    if [ ! -f "${pid_file}" ]; then
        echo "${name}: already stopped"
        return 0
    fi

    local pid
    pid="$(tr -d '[:space:]' < "${pid_file}")"
    if [ -z "${pid}" ] || ! process_matches "${pid}" "${marker}"; then
        echo "${name}: stale PID file removed; no matching process killed"
        rm -f -- "${pid_file}"
        return 0
    fi

    kill -TERM "${pid}"
    local attempt
    for attempt in $(seq 1 20); do
        if ! process_matches "${pid}" "${marker}"; then
            rm -f -- "${pid_file}"
            echo "${name}: STOPPED pid=${pid}"
            return 0
        fi
        sleep 0.25
    done
    echo "${name}: did not stop after SIGTERM pid=${pid}"
    return 1
}

start_all() {
    if [ ! -x "${PYTHON_BIN}" ]; then
        echo "Python not found: ${PYTHON_BIN}"
        return 1
    fi
    mkdir -p -- "${RUN_DIR}" "${LOG_DIR}"
    start_component "Raw Consumer" "${CONSUMER_PID_FILE}" "${CONSUMER_SCRIPT}" "${CONSUMER_LOG}" "${CONSUMER_SCRIPT}" || return 1
    start_component "V5 NORMAL Producer" "${PRODUCER_PID_FILE}" "${PRODUCER_SCRIPT}" "${PRODUCER_LOG}" "${PRODUCER_SCRIPT}" --mode normal-live || return 1
    status_all
}

stop_all() {
    stop_component "V5 NORMAL Producer" "${PRODUCER_PID_FILE}" "${PRODUCER_SCRIPT}"
    stop_component "Raw Consumer" "${CONSUMER_PID_FILE}" "${CONSUMER_SCRIPT}"
}

status_all() {
    local result=0
    component_status "V5 NORMAL Producer" "${PRODUCER_PID_FILE}" "${PRODUCER_SCRIPT}" || result=1
    component_status "Raw Consumer" "${CONSUMER_PID_FILE}" "${CONSUMER_SCRIPT}" || result=1
    return "${result}"
}

case "${1:-status}" in
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    restart)
        stop_all
        start_all
        ;;
    status)
        status_all
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 2
        ;;
esac
