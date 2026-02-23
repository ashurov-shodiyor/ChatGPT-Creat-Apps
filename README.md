# Offline Stable Diffusion Desktop App (Python + Tkinter)

This project is a **desktop app** that generates images locally with **Stable Diffusion** and a simple **Tkinter** user interface.

- No cloud API required for generation.
- Select a local model folder.
- Enter prompt/settings.
- Generated images are automatically saved to an output folder.

## Requirements

- Python 3.10+
- A local Stable Diffusion model directory (downloaded ahead of time)
- Optional but recommended: NVIDIA GPU with CUDA

Install dependencies:

```bash
pip install -r requirements.txt
```

## Prepare a Local Model (one-time)

You need to have a model available on disk first. Example directory:

```text
models/stable-diffusion-v1-5/
```

The app loads with `local_files_only=True`, so it does not fetch model files from the internet during generation.

## Run

```bash
python app.py
```

## How to use

1. Click **Load Model** after selecting your local model path.
2. Choose an output folder for generated images.
3. Enter prompt (and optional negative prompt).
4. Pick settings like steps, CFG guidance, size, and optional seed.
5. Click **Generate and Save**.

Saved files are named like:

```text
sd_YYYYMMDD_HHMMSS.png
```

## Notes

- Width/Height should be multiples of 8.
- First model load can take time.
- CPU works but is much slower than GPU.
