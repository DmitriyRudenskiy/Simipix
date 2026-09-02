"""Pose (DW-Pose) + face detection: найти похожих людей и одинаковые позы.

DW-Pose (COCO, 17 ключевых точек, пакет ``dwpose`` сbundled ONNX-весами) даёт
позу человека; лицо вырезается детектором ``yolov8n-face``. Поза нормализуется
по боксу человека (инвариантно к размеру/положению), поэтому позы двух разных
людей сопоставимы. Лицо встраивается той же моделью, что и полное изображение
(config.MODEL), для поиска похожих людей.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import config

# COCO 17 ключевых точек, порядок как в DW-Pose
KEYPART_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# связи для отрисовки скелета (COCO)
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (7, 11), (9, 11), (8, 12), (10, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# ponytail: путь к весам лиц задаётся в config.py — переопределить через env
# IMGSIM_FACE_MODEL. Дефолт — models/, туда же кладутся и веса модели.
FACE_MODEL = config.FACE_MODEL


@dataclass
class Pose:
    """Нормализованная поза одного человека: kps [17,3] в [0,1] (x,y,conf)."""
    kps: np.ndarray = field(default_factory=lambda: np.full((17, 3), np.nan, dtype=np.float32))
    score: float = 0.0

    def vector(self) -> np.ndarray:
        # только точки с conf>0; пропущенные точки = 0 (вектор фикс. длины 51)
        arr = np.where(np.isfinite(self.kps), self.kps, 0.0).astype(np.float32)
        return arr.reshape(-1)  # 17*3

    def to_dict(self) -> dict:
        return {"kps": self.kps.tolist(), "score": float(self.score)}

    @classmethod
    def from_dict(cls, d: dict) -> "Pose":
        kps = d.get("kps") or []
        arr = np.array(kps, dtype=np.float32).reshape(-1, 3) if kps else np.full((17, 3), np.nan, dtype=np.float32)
        p = cls()
        p.kps = arr
        p.score = float(d.get("score", 0.0))
        return p


class PoseDetector:
    """Ленивый загрузчик DW-Pose + face (CPU-safe, потокобезопасный)."""

    def __init__(self) -> None:
        self._whole_model = None
        self._face_model = None
        self._lock = threading.Lock()

    def _load_whole(self):
        if self._whole_model is None:
            from dwpose import wholebody
            ck = os.path.join(os.path.dirname(__import__("dwpose").__file__), "ckpts", "yzd-v", "DWPose")
            with self._lock:
                if self._whole_model is None:
                    self._whole_model = wholebody.Wholebody(
                        det_model_path=os.path.join(ck, "yolox_l.onnx"),
                        pose_model_path=os.path.join(ck, "dw-ll_ucoco_384.onnx"),
                    )
        return self._whole_model

    def _load_face(self):
        if self._face_model is None:
            with self._lock:
                if self._face_model is None:
                    from ultralytics import YOLO
                    self._face_model = YOLO(FACE_MODEL)
        return self._face_model

    def detect_pose(self, img: Image.Image) -> Pose | None:
        """Вернуть лучшую позу (макс. total_score) или None, если людей нет."""
        try:
            whole = self._load_whole()
            res = whole(np.array(img.convert("RGB")))
            people = whole.format_result(res)
        except Exception:
            return None
        if not people:
            return None
        best: Pose | None = None
        for person in people:
            kps = _from_pose_result(person)
            if kps is None:
                continue
            if best is None or kps[1] > best.score:
                best = Pose(kps=kps[0], score=kps[1])
        return best

    def crop_face(self, img: Image.Image) -> Image.Image | None:
        """Вырезать самое большое лицо. Вернуть RGB PIL или None."""
        try:
            res = self._load_face()(img.convert("RGB"), verbose=False)[0]
        except Exception:
            return None
        if res.boxes is None or len(res.boxes.data) == 0:
            return None
        box = res.boxes.data[0]  # x1,y1,x2,y2,conf
        w, h = img.size
        x1, y1 = max(0.0, float(box[0])), max(0.0, float(box[1]))
        x2, y2 = min(float(w), float(box[2])), min(float(h), float(box[3]))
        if x2 <= x1 or y2 <= y1:
            return None
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        rw, rh = (x2 - x1) * 1.6, (y2 - y1) * 1.9
        return img.crop((max(0, int(cx - rw / 2)), max(0, int(cy - rh / 2)),
                         min(w, int(cx + rw / 2)), min(h, int(cy + rh / 2)))).convert("RGB")


def _from_pose_result(person) -> tuple[np.ndarray, float] | None:
    """DW-Pose PoseResult -> (kps [17,3] норм, score). conf в [0,1]."""
    kpts = person.body.keypoints  # список Keypoint(x,y,score,id), score уже в [0,1]
    if not kpts or len(kpts) < 17:
        return None
    out = np.full((17, 3), np.nan, dtype=np.float32)
    for kp in kpts:
        if kp is None or kp.id is None or kp.id >= 17:
            continue
        out[kp.id, 0] = kp.x
        out[kp.id, 1] = kp.y
        out[kp.id, 2] = float(min(1.0, max(0.0, kp.score)))
    valid = np.isfinite(out[:, 0])
    if valid.sum() < 3:
        return None
    xs, ys = out[:, 0], out[:, 1]
    minx, maxx = xs[valid].min(), xs[valid].max()
    miny, maxy = ys[valid].min(), ys[valid].max()
    span_x = max(maxx - minx, 1e-6)
    span_y = max(maxy - miny, 1e-6)
    norm = out.copy()  # (17,3): x,y нормализуем, conf оставляем
    norm[:, 0] = (xs - minx) / span_x
    norm[:, 1] = (ys - miny) / span_y
    score = float(out[valid, 2].mean())
    return norm, score


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pose_similarity(p1: Pose | None, p2: Pose | None) -> float:
    if p1 is None or p2 is None:
        return 0.0
    return cosine(p1.vector(), p2.vector())
