# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85

## LIBERO-plus

LIBERO-plus is loaded in an isolated process so it does not replace the original editable LIBERO package used for
training. Set it up once (the asset archive is about 6.4 GB and supports resuming):

```bash
uv sync
bash examples/libero/setup_libero_plus.sh
```

Start the policy server in one terminal as usual. In a second terminal, run a small deterministic Objects Layout
pilot on `libero_object`:

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
uv run python examples/libero/run_libero_plus.py \
  --args.task-suite-name libero_object \
  --args.object-condition 3d \
  --args.task-category "Objects Layout" \
  --args.max-tasks 20 \
  --args.num-trials-per-task 1 \
  --args.seed 7 \
  --args.video-out-path data/libero_plus/videos/object_3d
```

The wrapper defaults to the same category, task count, and one trial when those flags are omitted. Use identical
selection flags and seed for the baseline, changing only `--args.object-condition` and the output directory. Per-episode
results are written to `metrics.jsonl`. Some task definitions are missing in the upstream checkout; unavailable tasks
are skipped deterministically and reported in the log.
