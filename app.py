import os
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import torch
from diffusers import StableDiffusionPipeline


class StableDiffusionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Offline Stable Diffusion Generator")
        self.root.geometry("760x560")

        self.pipeline = None
        self.generating = False

        self.model_path_var = tk.StringVar(value="./models/stable-diffusion-v1-5")
        self.save_dir_var = tk.StringVar(value="./generated_images")
        self.prompt_var = tk.StringVar()
        self.negative_prompt_var = tk.StringVar()
        self.steps_var = tk.IntVar(value=30)
        self.guidance_var = tk.DoubleVar(value=7.5)
        self.width_var = tk.IntVar(value=512)
        self.height_var = tk.IntVar(value=512)
        self.seed_var = tk.StringVar(value="")

        self._build_ui()

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            container,
            text="Generate Images Locally (No Cloud API)",
            font=("Segoe UI", 14, "bold"),
        )
        title.pack(anchor=tk.W, pady=(0, 12))

        self._path_row(
            container,
            label="Local model path",
            variable=self.model_path_var,
            browse_command=self.browse_model,
            browse_text="Browse Folder",
        )
        self._path_row(
            container,
            label="Output folder",
            variable=self.save_dir_var,
            browse_command=self.browse_output,
            browse_text="Choose Folder",
        )

        form = ttk.Frame(container)
        form.pack(fill=tk.X, pady=8)

        ttk.Label(form, text="Prompt").grid(row=0, column=0, sticky=tk.W, pady=(0, 4))
        ttk.Entry(form, textvariable=self.prompt_var).grid(
            row=1, column=0, columnspan=4, sticky=tk.EW, pady=(0, 8)
        )

        ttk.Label(form, text="Negative prompt").grid(
            row=2, column=0, sticky=tk.W, pady=(0, 4)
        )
        ttk.Entry(form, textvariable=self.negative_prompt_var).grid(
            row=3, column=0, columnspan=4, sticky=tk.EW, pady=(0, 8)
        )

        self._spin(form, "Steps", self.steps_var, 4, 0, from_=5, to=150)
        self._spin(form, "Guidance", self.guidance_var, 4, 1, from_=1.0, to=20.0, inc=0.5)
        self._spin(form, "Width", self.width_var, 4, 2, from_=256, to=1024, inc=64)
        self._spin(form, "Height", self.height_var, 4, 3, from_=256, to=1024, inc=64)

        ttk.Label(form, text="Seed (optional)").grid(row=6, column=0, sticky=tk.W, pady=(8, 4))
        ttk.Entry(form, textvariable=self.seed_var).grid(row=7, column=0, sticky=tk.W)

        for col in range(4):
            form.columnconfigure(col, weight=1)

        actions = ttk.Frame(container)
        actions.pack(fill=tk.X, pady=(16, 10))

        self.load_button = ttk.Button(actions, text="Load Model", command=self.load_model)
        self.load_button.pack(side=tk.LEFT)

        self.generate_button = ttk.Button(
            actions,
            text="Generate and Save",
            command=self.generate_image,
            state=tk.DISABLED,
        )
        self.generate_button.pack(side=tk.LEFT, padx=(8, 0))

        self.status_var = tk.StringVar(value="Load a local Stable Diffusion model to begin.")
        status = ttk.Label(container, textvariable=self.status_var, foreground="#125")
        status.pack(fill=tk.X)

        tips = ttk.Label(
            container,
            text=(
                "Tip: Download a model once (e.g. with Hugging Face CLI), then point\n"
                "the app to that local folder so generation works offline."
            ),
            foreground="#444",
        )
        tips.pack(anchor=tk.W, pady=(12, 0))

    def _path_row(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        browse_command,
        browse_text: str,
    ) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=4)

        ttk.Label(frame, text=label).pack(anchor=tk.W)

        inner = ttk.Frame(frame)
        inner.pack(fill=tk.X, pady=(2, 0))

        ttk.Entry(inner, textvariable=variable).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(inner, text=browse_text, command=browse_command).pack(side=tk.LEFT, padx=(8, 0))

    def _spin(self, parent, label, variable, row, column, from_, to, inc=1):
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky=tk.W, pady=(0, 4))
        ttk.Spinbox(
            parent,
            textvariable=variable,
            from_=from_,
            to=to,
            increment=inc,
            width=10,
        ).grid(row=row + 1, column=column, sticky=tk.W)

    def browse_model(self) -> None:
        folder = filedialog.askdirectory(title="Choose local model folder")
        if folder:
            self.model_path_var.set(folder)

    def browse_output(self) -> None:
        folder = filedialog.askdirectory(title="Choose save folder")
        if folder:
            self.save_dir_var.set(folder)

    def load_model(self) -> None:
        model_path = Path(self.model_path_var.get()).expanduser().resolve()
        if not model_path.exists():
            messagebox.showerror("Model not found", f"Path does not exist:\n{model_path}")
            return

        self.status_var.set("Loading model into memory... this can take a while.")
        self.load_button.configure(state=tk.DISABLED)

        def worker():
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                pipe = StableDiffusionPipeline.from_pretrained(
                    str(model_path),
                    torch_dtype=dtype,
                    safety_checker=None,
                    local_files_only=True,
                )
                pipe = pipe.to(device)
                if device == "cuda":
                    pipe.enable_attention_slicing()

                self.pipeline = pipe
                self.root.after(
                    0,
                    lambda: self._set_ready(
                        f"Model loaded on {device.upper()}. Ready to generate images."
                    ),
                )
            except Exception as exc:
                self.root.after(0, lambda: self._set_load_error(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _set_ready(self, text: str) -> None:
        self.status_var.set(text)
        self.generate_button.configure(state=tk.NORMAL)
        self.load_button.configure(state=tk.NORMAL)

    def _set_load_error(self, exc: Exception) -> None:
        self.status_var.set("Failed to load model.")
        self.load_button.configure(state=tk.NORMAL)
        messagebox.showerror("Load error", str(exc))

    def generate_image(self) -> None:
        if self.generating:
            return
        if self.pipeline is None:
            messagebox.showwarning("Model not loaded", "Load a model first.")
            return

        prompt = self.prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("Prompt required", "Please enter a prompt.")
            return

        width = self.width_var.get()
        height = self.height_var.get()
        if width % 8 != 0 or height % 8 != 0:
            messagebox.showwarning(
                "Invalid size", "Width and height should be multiples of 8."
            )
            return

        save_dir = Path(self.save_dir_var.get()).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

        self.generating = True
        self.generate_button.configure(state=tk.DISABLED)
        self.status_var.set("Generating image...")

        def worker():
            try:
                seed_text = self.seed_var.get().strip()
                generator = None
                if seed_text:
                    seed = int(seed_text)
                    generator = torch.Generator(device=self.pipeline.device).manual_seed(seed)

                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=self.negative_prompt_var.get().strip() or None,
                    num_inference_steps=int(self.steps_var.get()),
                    guidance_scale=float(self.guidance_var.get()),
                    width=width,
                    height=height,
                    generator=generator,
                )
                image = result.images[0]

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"sd_{timestamp}.png"
                output_path = save_dir / filename
                image.save(output_path)

                self.root.after(0, lambda: self._on_generate_success(output_path))
            except Exception as exc:
                self.root.after(0, lambda: self._on_generate_error(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_generate_success(self, output_path: Path) -> None:
        self.generating = False
        self.generate_button.configure(state=tk.NORMAL)
        self.status_var.set(f"Saved image: {output_path}")
        messagebox.showinfo("Done", f"Image saved to:\n{output_path}")

    def _on_generate_error(self, exc: Exception) -> None:
        self.generating = False
        self.generate_button.configure(state=tk.NORMAL)
        self.status_var.set("Image generation failed.")
        messagebox.showerror("Generation error", str(exc))


if __name__ == "__main__":
    os.makedirs("generated_images", exist_ok=True)
    root = tk.Tk()
    app = StableDiffusionApp(root)
    root.mainloop()
