#!/usr/bin/env bash
# Make sure recent Nsight Systems (nsys) and Nsight Compute (ncu) are available, and report whether ncu may read
# GPU performance counters. Safe to re-run; meant to be sourced so PATH is exported:
#
#   source scripts/setup_profilers.sh
#
# Blackwell (sm_120, our RTX PRO 5000) needs Nsight Compute / Systems >= 2025.1. Order of preference:
#   1. a system install that is new enough (/usr/local/cuda/bin, /opt/nvidia/...)
#   2. a previous user-space install in $NSIGHT_HOME (default: <project>/../.local/nsight, i.e. our team folder)
#   3. root: apt install from NVIDIA's CUDA repo
#   4. no root: download NVIDIA's .deb packages (or use ones we pushed into third_party/nsight/) and unpack them
#      with `dpkg -x` into $NSIGHT_HOME (no root needed). ~390 MB (ncu) + ~450 MB (nsys).
# NSIGHT_FORCE_USER_INSTALL=1 forces step 4 (used to test it on Colab).
_PROF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NSIGHT_HOME="${NSIGHT_HOME:-$(dirname "$_PROF_ROOT")/.local/nsight}"
NSIGHT_MIN="${NSIGHT_MIN:-2025.1}"

_addpath() { if [[ -n "$1" && -d "$1" && ":$PATH:" != *":$1:"* ]]; then export PATH="$1:$PATH"; fi; }
_ver() {  # first YYYY.N in a tool's --version output, e.g. 2025.3
    "$1" --version 2>/dev/null | grep -oE '20[0-9]{2}\.[0-9]+' | head -1
}
_ok() {  # tool name -> 0 if present and >= NSIGHT_MIN
    command -v "$1" >/dev/null || return 1
    local v; v="$(_ver "$1")"; [[ -n "$v" ]] || return 1
    [[ "$(printf '%s\n%s\n' "$NSIGHT_MIN" "$v" | sort -V | head -1)" == "$NSIGHT_MIN" ]]
}
_user_dirs() {  # PATH entries of a user-space install
    _addpath "$(ls -d "$NSIGHT_HOME"/opt/nvidia/nsight-compute/*/ 2>/dev/null | sort -V | tail -1 || true)"
    _addpath "$(ls -d "$NSIGHT_HOME"/opt/nvidia/nsight-systems/*/bin 2>/dev/null | sort -V | tail -1 || true)"
}
_user_install() {  # $1 = package prefix (nsight-compute | nsight-systems)
    local pkg="$1" distro deb url idx
    distro="ubuntu$(. /etc/os-release 2>/dev/null; echo "${VERSION_ID//./}")"
    case "$distro" in ubuntu2004|ubuntu2204|ubuntu2404) ;; *) distro=ubuntu2204 ;; esac
    mkdir -p "$NSIGHT_HOME/debs"
    deb="$(ls "$_PROF_ROOT"/third_party/nsight/${pkg}-20*.deb "$NSIGHT_HOME"/debs/${pkg}-20*.deb 2>/dev/null | sort -V | tail -1 || true)"
    if [[ -z "$deb" ]]; then
        url="https://developer.download.nvidia.com/compute/cuda/repos/$distro/x86_64"
        idx="$(curl -s -m 30 "$url/")"
        deb="$(grep -oE "${pkg}-20[0-9.]+_[^\"'<>]*_amd64\.deb" <<< "$idx" | sort -uV | tail -1)"
        if [[ -z "$deb" ]]; then echo "[profilers] cannot list $url (no internet?)"; return 1; fi
        echo "[profilers] downloading $deb"
        curl -sfL -m 900 -o "$NSIGHT_HOME/debs/$deb" "$url/$deb" || { echo "[profilers] download failed"; return 1; }
        deb="$NSIGHT_HOME/debs/$deb"
    fi
    echo "[profilers] unpacking $(basename "$deb") into $NSIGHT_HOME (no root needed)"
    if command -v dpkg >/dev/null; then dpkg -x "$deb" "$NSIGHT_HOME"
    else (cd "$NSIGHT_HOME" && ar x "$deb" data.tar.xz && tar -xJf data.tar.xz && rm -f data.tar.xz); fi
}

# 1 + 2: system and existing user-space installs
_addpath "$(ls -d /usr/local/cuda/bin 2>/dev/null || true)"
_addpath "$(ls -d /opt/nvidia/nsight-compute/*/ 2>/dev/null | sort -V | tail -1 || true)"
_addpath "$(ls -d /opt/nvidia/nsight-systems*/*/bin /opt/nvidia/nsight-systems/*/bin 2>/dev/null | sort -V | tail -1 || true)"
_user_dirs

for tool in ncu nsys; do
    pkg=$([[ $tool == ncu ]] && echo nsight-compute || echo nsight-systems)
    if [[ "${NSIGHT_FORCE_USER_INSTALL:-0}" != "1" ]] && _ok "$tool"; then continue; fi
    echo "[profilers] $tool missing or older than $NSIGHT_MIN ($(_ver "$tool" || true)); installing"
    if [[ "${NSIGHT_FORCE_USER_INSTALL:-0}" != "1" ]] && { [[ $EUID -eq 0 ]] || sudo -n true 2>/dev/null; }; then
        SUDO=""; if [[ $EUID -ne 0 ]]; then SUDO="sudo -n"; fi
        $SUDO apt-get update -qq >/dev/null 2>&1 || true
        $SUDO apt-get install -y -qq "$(apt-cache search --names-only "^${pkg}-20" | sort -V | tail -1 | cut -d' ' -f1)" \
            >/dev/null 2>&1 || echo "[profilers] apt install of $pkg failed"
        _addpath "$(ls -d /opt/nvidia/nsight-compute/*/ 2>/dev/null | sort -V | tail -1 || true)"
        _addpath "$(ls -d /opt/nvidia/nsight-systems*/*/bin /opt/nvidia/nsight-systems/*/bin 2>/dev/null | sort -V | tail -1 || true)"
    fi
    if [[ "${NSIGHT_FORCE_USER_INSTALL:-0}" == "1" ]] || ! _ok "$tool"; then
        _user_install "$pkg" && _user_dirs
    fi
done

echo "[profilers] nsys: $(command -v nsys || echo MISSING) $(_ver nsys)"
echo "[profilers] ncu : $(command -v ncu || echo MISSING) $(_ver ncu)"
if ! _ok ncu || ! _ok nsys; then echo "[profilers] WARNING: profiler older than $NSIGHT_MIN or missing (Blackwell needs >= 2025.1)"; fi

# Performance-counter permission (ERR_NVGPUCTRPERM => ncu unusable without root / driver setting)
if [[ -r /proc/driver/nvidia/params ]]; then
    echo "[profilers] driver $(grep -E 'RmProfilingAdminOnly' /proc/driver/nvidia/params || echo 'RmProfilingAdminOnly: n/a') (1 = only root may profile)"
fi
echo "[profilers] running as uid $EUID"
