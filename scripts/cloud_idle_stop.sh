#!/usr/bin/env bash
# Opt-in watchdog. Never start this on rai/hpg; run on the rented box only.
set -Eeuo pipefail

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }
usage() {
    printf 'Usage: %s MINUTES [--dry-run]\n' "${0##*/}"
    printf 'Stop after MINUTES without a Python GPU process. Poll every 30 seconds.\n'
    printf 'VOLUME_ROOT defaults to /workspace; --dry-run logs the stop and exits.\n'
}
if [[ ${1:-} == --help ]]; then usage; exit 0; fi
[[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }
[[ $1 =~ ^[1-9][0-9]{0,5}$ ]] || die 'MINUTES must be an integer in 1..999999.'
idle_seconds=$((10#$1 * 60))
dry_run=false
if [[ $# == 2 ]]; then
    [[ $2 == --dry-run ]] || die 'Only --dry-run is accepted as the second argument.'
    dry_run=true
fi
for tool in nvidia-smi timeout flock; do
    command -v "$tool" >/dev/null || die "Missing command: $tool"
done
VOLUME_ROOT=${VOLUME_ROOT:-/workspace}
[[ $VOLUME_ROOT == /* && $VOLUME_ROOT != / && -d $VOLUME_ROOT && -w $VOLUME_ROOT ]] ||
    die 'VOLUME_ROOT must be a writable persistent-volume directory other than /.'
mkdir -p -- "$VOLUME_ROOT/logs"
exec > >(tee -a "$VOLUME_ROOT/logs/cloud_idle_stop.log") 2>&1
exec 9>"$VOLUME_ROOT/.cloud-idle-stop.lock"
flock -n 9 || die 'An idle watchdog is already running on this volume.'
trap 'log "Watchdog stopped; no shutdown requested."; exit 0' INT TERM

stop_box() {
    if [[ -n ${RUNPOD_POD_ID:-} ]] && command -v runpodctl >/dev/null; then
        log "Idle threshold reached; stopping RunPod pod $RUNPOD_POD_ID."
        if $dry_run; then log 'DRY RUN: would run runpodctl stop pod RUNPOD_POD_ID.'; return; fi
        runpodctl stop pod "$RUNPOD_POD_ID"
    else
        log 'Idle threshold reached; requesting shutdown -h now.'
        if $dry_run; then log 'DRY RUN: would run shutdown -h now.'; return; fi
        if ((EUID == 0)); then
            shutdown -h now
        else
            sudo -n shutdown -h now
        fi
    fi
}
if ! $dry_run && { [[ -z ${RUNPOD_POD_ID:-} ]] || ! command -v runpodctl >/dev/null; }; then
    command -v shutdown >/dev/null || die 'shutdown is unavailable; configure runpodctl for a RunPod container.'
    if ((EUID != 0)); then
        command -v sudo >/dev/null && sudo -n true || die 'Run as root or provide passwordless sudo for shutdown.'
    fi
fi

# Return 0 for busy, 1 for positively idle, 2 for unknown. Inspect all visible
# compute processes, independent of CUDA_VISIBLE_DEVICES and GPU utilization.
# vLLM changes Python process titles; /proc/PID/exe still identifies Python.
gpu_activity() {
    local devices rows pid name exe comm
    devices=$(timeout 15 nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits 2>/dev/null) || return 2
    [[ $devices == *GPU-* ]] || return 2
    rows=$(timeout 15 nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null) || return 2
    while IFS=, read -r pid name; do
        pid=${pid//[[:space:]]/}
        [[ -n $pid ]] || continue
        [[ $pid =~ ^[0-9]+$ ]] || return 2
        name=${name,,}
        if [[ $name == *python* || $name == *vllm* ]]; then return 0; fi
        exe=$(readlink -- "/proc/$pid/exe" 2>/dev/null) || return 2
        comm=$(cat "/proc/$pid/comm" 2>/dev/null) || return 2
        if [[ ${exe,,} == *python* || ${comm,,} == *python* || ${comm,,} == *vllm* ]]; then return 0; fi
    done <<< "$rows"
    return 1
}

# /proc/uptime is monotonic across NTP clock changes. Restarting the helper
# begins a new grace period; there is no persisted/stale idle timestamp.
uptime_seconds() {
    local uptime unused
    read -r uptime unused < /proc/uptime
    printf '%s\n' "${uptime%%.*}"
}
idle_since=''
log "Watching GPU Python/vLLM processes; idle limit $1 minutes; dry-run=$dry_run; pid=$$."
while true; do
    status=0
    gpu_activity || status=$?
    now=$(uptime_seconds)
    case $status in
        0)
            [[ -z $idle_since ]] || log 'Python GPU activity resumed; resetting idle timer.'
            idle_since=''
            ;;
        1)
            if [[ -z $idle_since ]]; then
                idle_since=$now
                log 'No Python GPU process; starting idle timer.'
            fi
            if ((now - idle_since >= idle_seconds)); then
                # Recheck immediately before the destructive action.
                status=0
                gpu_activity || status=$?
                if ((status == 1)); then
                    stop_box || die 'Stop command failed; watchdog exiting without claiming the box stopped.'
                    exit 0
                fi
                idle_since=''
                log 'Final GPU check was busy or unknown; resetting idle timer.'
            fi
            ;;
        *)
            idle_since=''
            log 'GPU process status unavailable; resetting idle timer and keeping the box running.'
            ;;
    esac
    sleep 30
done
