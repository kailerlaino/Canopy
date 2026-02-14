#!/bin/bash
# Centralized cleanup for all Orca-related processes. Use between traces/runs
# so that leftover processes do not pile up (e.g. learner, actor, mahimahi).
# Source this file to get kill_orca_processes; or run the script directly.

kill_orca_processes() {
    echo "[kill_orca] Stopping Orca-related processes ..."
    # SIGTERM first for graceful shutdown
    sudo killall -s15 python client orca-server-mahimahi orca-server-mahimahi_v0 orca-server-mahimahi_v2 mm-link mm-delay 2>/dev/null || true
    sleep 2
    # SIGKILL to ensure nothing is left
    sudo killall -s9 python client orca-server-mahimahi orca-server-mahimahi_v0 orca-server-mahimahi_v2 mm-link mm-delay 2>/dev/null || true
    # Free port 44444 in case something is still bound (e.g. previous run)
    if command -v lsof &>/dev/null; then
        pids=$(sudo lsof -i :44444 2>/dev/null | tr -s ' ' | cut -d ' ' -f2 | grep -v PID)
        if [[ -n "$pids" ]]; then
            echo "$pids" | xargs sudo kill -9 2>/dev/null || true
        fi
    fi
    echo "[kill_orca] Done."
}

# Run when script is executed (not when sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    kill_orca_processes
fi
