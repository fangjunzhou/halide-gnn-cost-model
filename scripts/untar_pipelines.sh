#!/usr/bin/env bash

set -euo pipefail

usage() {
	local exit_code=${1:-0}
	cat <<'USAGE'
Usage: untar_pipelines.sh [--source DIR] [--dest DIR] [--jobs N]

Extracts every .tar archive from the source directory (defaults to pwd)
into the destination directory (defaults to source) using up to N
parallel extraction jobs (defaults to detected CPU count).
USAGE
	exit "$exit_code"
}

source_dir="$PWD"
dest_dir=""
jobs=""

while [[ $# -gt 0 ]]; do
	case "$1" in
		-s|--source)
			[[ $# -ge 2 ]] || usage 1
			source_dir="$2"
			shift 2
			;;
		-d|--dest)
			[[ $# -ge 2 ]] || usage 1
			dest_dir="$2"
			shift 2
			;;
		-j|--jobs)
			[[ $# -ge 2 ]] || usage 1
			jobs="$2"
			shift 2
			;;
		-h|--help)
			usage 0
			;;
		*)
			echo "Unknown argument: $1" >&2
			usage 1
			;;
	esac
done

if [[ -z "$dest_dir" ]]; then
	dest_dir="$source_dir"
fi

if [[ ! -d "$source_dir" ]]; then
	echo "Source directory does not exist: $source_dir" >&2
	exit 1
fi

mkdir -p "$dest_dir"

detect_cpu_count() {
	if command -v nproc >/dev/null 2>&1; then
		nproc
	elif [[ "$(uname -s)" == "Darwin" ]]; then
		sysctl -n hw.logicalcpu
	else
		getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1
	fi
}

if [[ -z "$jobs" ]]; then
	jobs="$(detect_cpu_count)"
fi

if ! [[ "$jobs" =~ ^[0-9]+$ ]] || (( jobs < 1 )); then
	echo "Invalid job count: $jobs" >&2
	exit 1
fi

echo "Using $jobs parallel extraction jobs"

mapfile -t tarballs < <(find "$source_dir" -maxdepth 1 -type f -name '*.tar' | sort)

if [[ ${#tarballs[@]} -eq 0 ]]; then
	echo "No .tar archives found in $source_dir"
	exit 0
fi

running_pids=()
failure=0

# Launch tar extractions with a bounded pool of background jobs.

start_job() {
	local tarball="$1"
	(
		set -euo pipefail
		echo "Extracting $tarball -> $dest_dir"
		tar -xvf "$tarball" -C "$dest_dir"
	) &
	running_pids+=("$!")
}

wait_for_oldest() {
	local pid="$1"
	if ! wait "$pid"; then
		failure=1
	fi
}

for tarball in "${tarballs[@]}"; do
	start_job "$tarball"
	if (( ${#running_pids[@]} >= jobs )); then
		wait_for_oldest "${running_pids[0]}"
		running_pids=("${running_pids[@]:1}")
		if (( failure )); then
			echo "Extraction failed; stopping new jobs" >&2
			break
		fi
	fi
done

for pid in "${running_pids[@]}"; do
	wait_for_oldest "$pid"
done

if (( failure )); then
	echo "One or more extractions failed" >&2
	exit 1
fi
