"""Обёртка над facebook/dinov2-giant для получения эмбеддингов.

Всегда используется самая большая модель (giant, 1536-dim) — максимальное
качество поиска. Веса можно загрузить локально (models/) — для этого сначала
выполните `python download_model.py`, тогда офлайн-режим работает без Hub.

Эмбеддинг = CLS-токен последнего слоя, L2-нормализованный. На нормализованных
векторах косинусная близость эквивалентна скалярному произведению и
корректно работает с метрикой cosine в LanceDB.
"""

from pathlib import Path

import numpy as np
import torch

from . import config


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


class DINOv2Embedder:
    def __init__(self, model_dir: str | Path | None = None,
                 device: str | None = None, log=print):
        # Всегда самая большая модель — без выбора варианта.
        self.variant = config.MODEL
        self.model_id = config.MODEL_ID
        self.dim = config.MODEL_DIM
        self.model_dir = Path(model_dir).expanduser() if model_dir else None
        self.device = pick_device(device)
        # fp16 только на CUDA; на CPU — fp32
        self.half = self.device.startswith("cuda")

        from transformers import AutoImageProcessor, AutoModel

        # Локальные веса优先, иначе — с HuggingFace Hub (потребуется загрузка).
        source = self.model_dir if (self.model_dir and self.model_dir.is_dir()) \
            else self.model_id
        try:
            self.processor = AutoImageProcessor.from_pretrained(source)
        except Exception:
            # резерв: медленный PIL-процессор (неrequires torchvision)
            self.processor = AutoImageProcessor.from_pretrained(
                source, use_fast=False)

        if self.half:
            try:
                self.model = AutoModel.from_pretrained(
                    source, dtype=torch.float16)
            except TypeError:
                try:
                    self.model = AutoModel.from_pretrained(
                        source, torch_dtype=torch.float16)
                except TypeError:
                    self.model = AutoModel.from_pretrained(source).half()
        else:
            self.model = AutoModel.from_pretrained(source)

        self.model.to(self.device)
        self.model.eval()
        if log:
            log(f"Модель: {self.model_id} ({self.dim}-dim) | "
                f"устройство: {self.device} | "
                f"точность: {'fp16' if self.half else 'fp32'} | "
                f"источник: {source}")

    @torch.inference_mode()
    def embed_images(self, images: list) -> np.ndarray:
        """Список PIL-изображений -> матрица (N, dim) float32, L2-нормализована."""
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        inputs = self.processor(images=images, return_tensors="pt")
        pv = inputs["pixel_values"].to(self.device)
        if self.half:
            pv = pv.half()
        out = self.model(pixel_values=pv).last_hidden_state[:, 0]
        emb = torch.nn.functional.normalize(out, dim=-1)
        return emb.float().cpu().numpy()

    def embed_with_retry(self, images: list, log=print) -> np.ndarray:
        """Эмбеддинг батча с автопонижением размера при нехватке памяти."""
        try:
            return self.embed_images(images)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
        except MemoryError:
            pass
        if len(images) == 1:
            raise
        mid = len(images) // 2
        log(f"  Не хватило памяти: уменьшаю батч до {mid}")
        a = self.embed_with_retry(images[:mid], log=log)
        b = self.embed_with_retry(images[mid:], log=log)
        return np.concatenate([a, b], axis=0)
