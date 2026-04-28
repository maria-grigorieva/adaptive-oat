#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${ROOT_DIR}/data/libero"
HDF5_DIR="${DATA_ROOT}/hdf5_datasets"
TARGET_ZARR="${DATA_ROOT}/libero_spatial_tiny.zarr"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

mkdir -p "${HDF5_DIR}"

find_downloader() {
  local local_script="${ROOT_DIR}/third_party/LIBERO/benchmark_scripts/download_libero_datasets.py"
  if [[ -f "${local_script}" ]]; then
    printf '%s\n' "${local_script}"
    return 0
  fi

  "${PYTHON_BIN}" - <<'PY'
import importlib.util
import pathlib

spec = importlib.util.find_spec("libero")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit(1)

roots = [pathlib.Path(p) for p in spec.submodule_search_locations]
candidates = []
for root in roots:
    candidates.append(root / "benchmark_scripts" / "download_libero_datasets.py")
    candidates.append(root.parent / "benchmark_scripts" / "download_libero_datasets.py")

for candidate in candidates:
    if candidate.is_file():
        print(candidate)
        raise SystemExit(0)

raise SystemExit(1)
PY
}

DOWNLOADER=""
if DOWNLOADER="$(find_downloader 2>/dev/null)"; then
  echo "Using LIBERO downloader: ${DOWNLOADER}"
else
  cat <<EOF
Could not find a LIBERO dataset downloader.

Expected one of:
  - ${ROOT_DIR}/third_party/LIBERO/benchmark_scripts/download_libero_datasets.py
  - an installed LIBERO package that ships benchmark_scripts/download_libero_datasets.py

To fix this, either:
  1. initialize the submodule:
     git submodule update --init --recursive
  2. or install LIBERO into the active Python environment

Nothing was downloaded.
EOF
  exit 1
fi

echo "Using Python interpreter: ${PYTHON_BIN}"
if compgen -G "${HDF5_DIR}/libero_spatial/*_demo.hdf5" > /dev/null; then
  echo "Reusing existing libero_spatial download under ${HDF5_DIR}/libero_spatial"
else
  echo "Downloading only libero_spatial into ${HDF5_DIR}"
  find "${HDF5_DIR}" -maxdepth 1 -type f -name '*.zip' -delete
  printf 'y\n' | "${PYTHON_BIN}" "${DOWNLOADER}" \
    --download-dir "${HDF5_DIR}" \
    --datasets libero_spatial
fi

if [[ -d "${HDF5_DIR}/libero_spatial" ]]; then
  echo "Flattening downloaded libero_spatial HDF5 files into ${HDF5_DIR}"
  find "${HDF5_DIR}/libero_spatial" -maxdepth 1 -type f -name '*_demo.hdf5' -print0 | while IFS= read -r -d '' hdf5_path; do
    cp -f "${hdf5_path}" "${HDF5_DIR}/$(basename "${hdf5_path}")"
  done
fi

echo "Converting HDF5 files with --num_sample_demo 1"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/convert_libero_dataset.py" \
  --root_dir "${DATA_ROOT}" \
  --hdf5_dir_name hdf5_datasets \
  --num_sample_demo 1

mapfile -t SPATIAL_TASKS < <("${PYTHON_BIN}" - <<'PY' 2>/dev/null || true
from libero.libero.benchmark.libero_suite_task_map import libero_task_map
for task_name in libero_task_map["libero_spatial"]:
    print(task_name)
PY
)

if [[ "${#SPATIAL_TASKS[@]}" -gt 0 ]]; then
  HDF5_FILES=()
  for task_name in "${SPATIAL_TASKS[@]}"; do
    hdf5_path="${HDF5_DIR}/${task_name}_demo.hdf5"
    if [[ -f "${hdf5_path}" ]]; then
      HDF5_FILES+=("${hdf5_path}")
    fi
  done
else
  echo "Warning: could not import libero_task_map, so the helper will merge every *_demo.hdf5 file in ${HDF5_DIR}." >&2
  mapfile -t HDF5_FILES < <(find "${HDF5_DIR}" -maxdepth 1 -type f -name '*_demo.hdf5' | sort)
fi

if [[ "${#HDF5_FILES[@]}" -eq 0 ]]; then
  echo "No *_demo.hdf5 files were found in ${HDF5_DIR} after download." >&2
  exit 1
fi

ZARR_ARGS=()
for hdf5_path in "${HDF5_FILES[@]}"; do
  task_name="$(basename "${hdf5_path}" "_demo.hdf5")"
  zarr_path="${DATA_ROOT}/${task_name}_N1.zarr"
  if [[ ! -d "${zarr_path}" ]]; then
    echo "Expected converted dataset ${zarr_path} was not found." >&2
    exit 1
  fi
  ZARR_ARGS+=(-p "${zarr_path}")
done

rm -rf "${TARGET_ZARR}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/merge_data.py" \
  "${ZARR_ARGS[@]}" \
  -s "${TARGET_ZARR}" \
  --shuffle

echo
echo "Prepared tiny LIBERO dataset artifacts:"
for hdf5_path in "${HDF5_FILES[@]}"; do
  du -sh "${hdf5_path}"
done
for zarr_path in "${DATA_ROOT}"/*_N1.zarr "${TARGET_ZARR}"; do
  if [[ -e "${zarr_path}" ]]; then
    du -sh "${zarr_path}"
  fi
done
