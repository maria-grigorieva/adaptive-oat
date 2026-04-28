#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${ROOT_DIR}/data/libero"
HDF5_DIR="${DATA_ROOT}/hdf5_datasets"
TARGET_ZARR="${DATA_ROOT}/libero_spatial_tinyplus.zarr"
NUM_SAMPLE_DEMO="${NUM_SAMPLE_DEMO:-2}"
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
  echo "Could not find a LIBERO dataset downloader." >&2
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

echo "Converting HDF5 files with --num_sample_demo ${NUM_SAMPLE_DEMO}"
DATA_ROOT_ENV="${DATA_ROOT}" \
HDF5_DIR_ENV="${HDF5_DIR}" \
TARGET_ZARR_ENV="${TARGET_ZARR}" \
NUM_SAMPLE_DEMO_ENV="${NUM_SAMPLE_DEMO}" \
"${PYTHON_BIN}" - <<'PY'
import glob
import os
import pathlib
import zarr

from oat.env.libero.dataset_conversion import convert_libero_hdft_to_zarr
from oat.common.replay_buffer import ReplayBuffer

root = os.environ["DATA_ROOT_ENV"]
hdf5_dir = os.environ["HDF5_DIR_ENV"]
target_zarr = os.environ["TARGET_ZARR_ENV"]
num_sample_demo = int(os.environ["NUM_SAMPLE_DEMO_ENV"])

hdf5_paths = sorted(glob.glob(os.path.join(hdf5_dir, "*_demo.hdf5")))
if not hdf5_paths:
    raise SystemExit(f"No *_demo.hdf5 files found in {hdf5_dir}")

compressor = zarr.Blosc(cname="zstd", clevel=5, shuffle=1)
save_paths = []
for hdf5_path in hdf5_paths:
    task_name = pathlib.Path(hdf5_path).stem[:-len("_demo")]
    replay_buffer = convert_libero_hdft_to_zarr(
        hdf5_path=hdf5_path,
        sample_ndemo=num_sample_demo,
    )
    save_path = os.path.join(root, f"{task_name}_N{replay_buffer.n_episodes}.zarr")
    if os.path.exists(save_path):
        os.system(f'rm -rf "{save_path}"')
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    replay_buffer.save_to_path(save_path, compressor=compressor)
    save_paths.append(save_path)
    print(f"saved {save_path}")

merged = ReplayBuffer.create_empty_zarr()
for save_path in save_paths:
    buffer = ReplayBuffer.create_from_path(save_path)
    for episode_idx in range(buffer.n_episodes):
        merged.add_episode(buffer.get_episode(episode_idx, copy=True))

if os.path.exists(target_zarr):
    os.system(f'rm -rf "{target_zarr}"')
merged.save_to_path(target_zarr, compressors=compressor)

episode_count = int(merged.n_episodes)
action_steps = int(merged.n_steps)
print(f"TARGET_ZARR={target_zarr}")
print(f"EPISODES={episode_count}")
print(f"ACTION_STEPS={action_steps}")
PY

echo
echo "Prepared tinyplus LIBERO dataset artifacts:"
du -sh "${TARGET_ZARR}"
TARGET_ZARR_ENV="${TARGET_ZARR}" "${PYTHON_BIN}" - <<'PY'
import zarr
import os
path = os.environ["TARGET_ZARR_ENV"]
root = zarr.open(path, mode="r")
episode_count = int(root["meta"]["episode_ends"].shape[0])
action_steps = int(root["data"]["action"].shape[0])
print(f"episodes={episode_count}")
print(f"action_steps={action_steps}")
PY
