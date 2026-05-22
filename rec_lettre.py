from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import product
import math
import time

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
# Preprocessing
# -----------------------------

def binarize(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (image.astype(float) >= threshold).astype(np.uint8)


def denoise_majority(image: np.ndarray, min_neighbors: int = 2) -> np.ndarray:
    """Remove isolated pixels with a simple 8-neighborhood rule."""
    img = image.astype(np.uint8)
    h, w = img.shape
    out = img.copy()
    for x in range(h):
        for y in range(w):
            if img[x, y] == 0:
                continue
            neighbors = 0
            for nx in range(max(0, x - 1), min(h, x + 2)):
                for ny in range(max(0, y - 1), min(w, y + 2)):
                    if nx == x and ny == y:
                        continue
                    neighbors += int(img[nx, ny])
            if neighbors < min_neighbors:
                out[x, y] = 0
    return out


def bounding_box(image: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(image > 0)
    if len(coords) == 0:
        return 0, image.shape[0], 0, image.shape[1]
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    return int(x0), int(x1), int(y0), int(y1)


def crop_to_content(image: np.ndarray, padding: int = 0) -> np.ndarray:
    x0, x1, y0, y1 = bounding_box(image)
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(image.shape[0], x1 + padding)
    y1 = min(image.shape[1], y1 + padding)
    return image[x0:x1, y0:y1]


def resize_nearest(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    th, tw = shape
    h, w = image.shape
    if h == 0 or w == 0:
        return np.zeros(shape, dtype=image.dtype)
    out = np.zeros(shape, dtype=float)
    for x in range(th):
        sx = min(h - 1, int(round((x / max(1, th - 1)) * (h - 1))))
        for y in range(tw):
            sy = min(w - 1, int(round((y / max(1, tw - 1)) * (w - 1))))
            out[x, y] = image[sx, sy]
    return out


def center_on_canvas(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = np.zeros(shape, dtype=float)
    ih, iw = image.shape
    ox = max(0, (h - ih) // 2)
    oy = max(0, (w - iw) // 2)
    x1 = min(h, ox + ih)
    y1 = min(w, oy + iw)
    out[ox:x1, oy:y1] = image[: x1 - ox, : y1 - oy]
    return out


def affine_transform(image: np.ndarray, matrix: np.ndarray, output_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Inverse-mapped nearest-neighbor affine transform."""
    img = image.astype(float)
    h, w = img.shape
    if output_shape is None:
        output_shape = (h, w)
    oh, ow = output_shape

    out = np.zeros((oh, ow), dtype=float)
    inv = np.linalg.inv(matrix)

    cx_in, cy_in = (h - 1) / 2.0, (w - 1) / 2.0
    cx_out, cy_out = (oh - 1) / 2.0, (ow - 1) / 2.0

    for x in range(oh):
        for y in range(ow):
            v = np.array([x - cx_out, y - cy_out, 1.0])
            sx, sy, _ = inv @ v
            sx += cx_in
            sy += cy_in
            isx = int(round(sx))
            isy = int(round(sy))
            if 0 <= isx < h and 0 <= isy < w:
                out[x, y] = img[isx, isy]
    return out


def deskew_via_moments(image: np.ndarray) -> np.ndarray:
    """Light shear correction from second moments."""
    img = image.astype(float)
    pts = np.argwhere(img > 0)
    if len(pts) < 2:
        return img
    x = pts[:, 0].astype(float)
    y = pts[:, 1].astype(float)
    mx, my = x.mean(), y.mean()
    u20 = np.mean((x - mx) ** 2)
    u02 = np.mean((y - my) ** 2)
    u11 = np.mean((x - mx) * (y - my))
    denom = (u20 - u02)
    if abs(denom) < 1e-8:
        return img
    theta = 0.5 * math.atan2(2.0 * u11, denom)
    # Small-angle correction only.
    theta = float(np.clip(theta, -0.25, 0.25))
    c, s = math.cos(-theta), math.sin(-theta)
    mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    return affine_transform(img, mat, output_shape=img.shape)


def binary_erode(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.uint8)
    h, w = img.shape
    out = np.zeros_like(img)
    for x in range(1, h - 1):
        for y in range(1, w - 1):
            block = img[x - 1 : x + 2, y - 1 : y + 2]
            out[x, y] = 1 if np.all(block == 1) else 0
    return out


def binary_dilate(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.uint8)
    h, w = img.shape
    out = np.zeros_like(img)
    for x in range(h):
        for y in range(w):
            if img[x, y] == 1:
                for nx in range(max(0, x - 1), min(h, x + 2)):
                    for ny in range(max(0, y - 1), min(w, y + 2)):
                        out[nx, ny] = 1
    return out


def normalize_stroke_thickness(image: np.ndarray, target_density: float = 0.22) -> np.ndarray:
    img = image.astype(np.uint8)
    current = float(np.mean(img))
    if current <= 1e-8:
        return img
    if current > target_density * 1.3:
        return binary_erode(img)
    if current < target_density * 0.7:
        return binary_dilate(img)
    return img


def preprocess_image(
    image: np.ndarray,
    output_shape: tuple[int, int],
    threshold: float = 0.5,
) -> np.ndarray:
    img = binarize(image, threshold=threshold)
    img = denoise_majority(img)
    img = crop_to_content(img, padding=1)
    img = deskew_via_moments(img)
    img = crop_to_content(img, padding=1)
    img = resize_nearest(img, output_shape)
    img = normalize_stroke_thickness(img)
    img = center_on_canvas(img, output_shape)
    return img.astype(np.uint8)


# -----------------------------
# Skeleton & topology
# -----------------------------

def _neighbors8(padded: np.ndarray, x: int, y: int) -> list[int]:
    return [
        int(padded[x - 1, y]),
        int(padded[x - 1, y + 1]),
        int(padded[x, y + 1]),
        int(padded[x + 1, y + 1]),
        int(padded[x + 1, y]),
        int(padded[x + 1, y - 1]),
        int(padded[x, y - 1]),
        int(padded[x - 1, y - 1]),
    ]


def zhang_suen_thinning(image: np.ndarray) -> np.ndarray:
    """Classical thinning for binary images."""
    img = image.astype(np.uint8).copy()
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            to_remove: list[tuple[int, int]] = []
            padded = np.pad(img, 1, mode="constant")
            h, w = img.shape
            for x in range(1, h + 1):
                for y in range(1, w + 1):
                    if padded[x, y] != 1:
                        continue
                    n = _neighbors8(padded, x, y)
                    b = sum(n)
                    if b < 2 or b > 6:
                        continue
                    a = sum((n[i] == 0 and n[(i + 1) % 8] == 1) for i in range(8))
                    if a != 1:
                        continue
                    p2, p4, p6, p8 = n[0], n[2], n[4], n[6]
                    if step == 0:
                        if p2 * p4 * p6 != 0:
                            continue
                        if p4 * p6 * p8 != 0:
                            continue
                    else:
                        if p2 * p4 * p8 != 0:
                            continue
                        if p2 * p6 * p8 != 0:
                            continue
                    to_remove.append((x - 1, y - 1))
            if to_remove:
                changed = True
                for x, y in to_remove:
                    img[x, y] = 0
    return img


def _count_components(binary: np.ndarray, target: int) -> int:
    h, w = binary.shape
    visited = np.zeros((h, w), dtype=bool)
    count = 0
    for i in range(h):
        for j in range(w):
            if visited[i, j] or int(binary[i, j]) != target:
                continue
            count += 1
            q = deque([(i, j)])
            visited[i, j] = True
            while q:
                x, y = q.popleft()
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny] and int(binary[nx, ny]) == target:
                        visited[nx, ny] = True
                        q.append((nx, ny))
    return count


def _count_holes(binary: np.ndarray) -> int:
    h, w = binary.shape
    bg = (binary == 0).astype(np.uint8)
    visited = np.zeros((h, w), dtype=bool)
    q = deque()
    for i in range(h):
        for j in (0, w - 1):
            if bg[i, j] and not visited[i, j]:
                visited[i, j] = True
                q.append((i, j))
    for j in range(w):
        for i in (0, h - 1):
            if bg[i, j] and not visited[i, j]:
                visited[i, j] = True
                q.append((i, j))
    while q:
        x, y = q.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < h and 0 <= ny < w and bg[nx, ny] and not visited[nx, ny]:
                visited[nx, ny] = True
                q.append((nx, ny))
    holes = 0
    for i in range(h):
        for j in range(w):
            if bg[i, j] and not visited[i, j]:
                holes += 1
                q2 = deque([(i, j)])
                visited[i, j] = True
                while q2:
                    x, y = q2.popleft()
                    for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                        if 0 <= nx < h and 0 <= ny < w and bg[nx, ny] and not visited[nx, ny]:
                            visited[nx, ny] = True
                            q2.append((nx, ny))
    return holes


def skeleton_graph_stats(skeleton: np.ndarray) -> tuple[int, int, float]:
    endpoints = 0
    junctions = 0
    degrees = []
    h, w = skeleton.shape
    for x in range(h):
        for y in range(w):
            if skeleton[x, y] == 0:
                continue
            deg = 0
            for nx in range(max(0, x - 1), min(h, x + 2)):
                for ny in range(max(0, y - 1), min(w, y + 2)):
                    if nx == x and ny == y:
                        continue
                    deg += int(skeleton[nx, ny])
            degrees.append(deg)
            if deg == 1:
                endpoints += 1
            elif deg >= 3:
                junctions += 1
    mean_deg = float(np.mean(degrees)) if degrees else 0.0
    return endpoints, junctions, mean_deg


def topology_signature(image: np.ndarray) -> np.ndarray:
    binary = (image > 0).astype(np.uint8)
    components = _count_components(binary, 1)
    holes = _count_holes(binary)
    skeleton = zhang_suen_thinning(binary)
    endpoints, junctions, mean_deg = skeleton_graph_stats(skeleton)
    return np.array([components, holes, endpoints, junctions, mean_deg], dtype=float)


# -----------------------------
# CVM core (distances + gamma)
# -----------------------------

def point_distance(kind: str, x: int, y: int, u: int, v: int) -> float:
    dx = abs(x - u)
    dy = abs(y - v)
    if kind == "manhattan":
        return float(dx + dy)
    if kind == "euclidean":
        return float(math.sqrt(dx * dx + dy * dy))
    if kind == "chebyshev":
        return float(max(dx, dy))
    raise ValueError(f"unsupported distance kind: {kind}")


def geodesic_distance_map(skeleton: np.ndarray, source: tuple[int, int]) -> np.ndarray:
    """Geodesic distances along skeleton graph (inf if unreachable)."""
    h, w = skeleton.shape
    dist = np.full((h, w), np.inf, dtype=float)
    sx, sy = source
    if skeleton[sx, sy] == 0:
        return dist
    q = deque([(sx, sy)])
    dist[sx, sy] = 0.0
    while q:
        x, y = q.popleft()
        for nx in range(max(0, x - 1), min(h, x + 2)):
            for ny in range(max(0, y - 1), min(w, y + 2)):
                if nx == x and ny == y:
                    continue
                if skeleton[nx, ny] == 0:
                    continue
                nd = dist[x, y] + 1.0
                if nd < dist[nx, ny]:
                    dist[nx, ny] = nd
                    q.append((nx, ny))
    return dist


def compute_frequency_matrix(samples: list[np.ndarray]) -> np.ndarray:
    if not samples:
        raise ValueError("samples must contain at least one matrix")
    shape = samples[0].shape
    if any(s.shape != shape for s in samples):
        raise ValueError("all samples in a class must share the same shape")
    return np.stack(samples, axis=0).astype(float).mean(axis=0)


def learn_local_gamma_map(samples: list[np.ndarray], base_gamma: float = 0.85) -> np.ndarray:
    """Higher local variability -> higher gamma (more tolerance)."""
    stacked = np.stack(samples, axis=0).astype(float)
    freq = stacked.mean(axis=0)
    var = stacked.var(axis=0)
    if float(var.max()) <= 1e-10:
        norm = np.zeros_like(var)
    else:
        norm = var / float(var.max())
    gamma_map = base_gamma + 0.12 * norm
    return np.clip(gamma_map, 0.5, 0.98)


def diffuse_value_matrix(
    frequency: np.ndarray,
    gamma: float = 0.9,
    distance_kind: str = "manhattan",
    gamma_map: np.ndarray | None = None,
    geodesic_skeleton: np.ndarray | None = None,
) -> np.ndarray:
    if not (0 < gamma <= 1):
        raise ValueError("gamma must satisfy 0 < gamma <= 1")

    h, w = frequency.shape
    if distance_kind == "manhattan" and gamma_map is None:
        return diffuse_manhattan_separable(frequency, gamma)

    value = np.zeros((h, w), dtype=float)
    active = np.argwhere(frequency > 0)

    for u, v in active:
        weight = float(frequency[u, v])
        local_gamma = float(gamma_map[u, v]) if gamma_map is not None else gamma

        if distance_kind == "geodesic":
            if geodesic_skeleton is None:
                raise ValueError("geodesic_skeleton is required for geodesic distance")
            dist_map = geodesic_distance_map(geodesic_skeleton, (int(u), int(v)))
            valid = np.isfinite(dist_map)
            value[valid] += weight * np.power(local_gamma, dist_map[valid])
            continue

        for x in range(h):
            for y in range(w):
                d = point_distance(distance_kind, x, y, int(u), int(v))
                value[x, y] += weight * (local_gamma**d)
    return value


def _diffuse_line_separable(line: np.ndarray, gamma: float) -> np.ndarray:
    n = line.shape[0]
    left = np.zeros(n, dtype=float)
    right = np.zeros(n, dtype=float)
    left[0] = float(line[0])
    for i in range(1, n):
        left[i] = float(line[i]) + gamma * left[i - 1]
    right[n - 1] = float(line[n - 1])
    for i in range(n - 2, -1, -1):
        right[i] = float(line[i]) + gamma * right[i + 1]
    return left + right - line.astype(float)


def diffuse_manhattan_separable(frequency: np.ndarray, gamma: float) -> np.ndarray:
    """Exact Manhattan diffusion via 1D separable passes."""
    tmp = np.zeros_like(frequency, dtype=float)
    out = np.zeros_like(frequency, dtype=float)
    for i in range(frequency.shape[0]):
        tmp[i, :] = _diffuse_line_separable(frequency[i, :], gamma)
    for j in range(frequency.shape[1]):
        out[:, j] = _diffuse_line_separable(tmp[:, j], gamma)
    return out


def build_penalty_matrix(frequency: np.ndarray) -> np.ndarray:
    return 1.0 - frequency


def score_image(
    image: np.ndarray,
    value_matrix: np.ndarray,
    penalty_matrix: np.ndarray | None = None,
    lambda_penalty: float = 0.0,
    score_mode: str = "cosine",
) -> float:
    img = image.astype(float)
    positive = float(np.sum(img * value_matrix))
    negative = 0.0
    if penalty_matrix is not None and lambda_penalty > 0:
        negative = float(np.sum(img * penalty_matrix)) * lambda_penalty
    raw = positive - negative
    if score_mode == "raw":
        return raw
    if score_mode == "mass":
        denom = float(np.sum(img)) * float(np.sum(value_matrix))
        return 0.0 if denom <= 0 else raw / denom
    if score_mode == "cosine":
        norm_img = float(np.linalg.norm(img))
        norm_val = float(np.linalg.norm(value_matrix))
        denom = norm_img * norm_val
        return 0.0 if denom <= 0 else raw / denom
    raise ValueError(f"unsupported score mode: {score_mode}")


# -----------------------------
# Geometric invariance
# -----------------------------

@dataclass
class TransformConfig:
    translations: list[tuple[int, int]]
    rotations_deg: list[float]
    scales: list[float]
    shears: list[float]


def transform_matrix(rotation_deg: float, scale: float, shear: float, tx: float, ty: float) -> np.ndarray:
    th = math.radians(rotation_deg)
    c, s = math.cos(th), math.sin(th)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    sc = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=float)
    sh = np.array([[1, shear, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    tr = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=float)
    return tr @ sh @ sc @ rot


def transformed_variants(image: np.ndarray, cfg: TransformConfig) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for (dx, dy), rot, sc, sh in product(cfg.translations, cfg.rotations_deg, cfg.scales, cfg.shears):
        mat = transform_matrix(rot, sc, sh, dx, dy)
        variants.append((affine_transform(image, mat, output_shape=image.shape) > 0.5).astype(np.uint8))
    return variants


def select_transform_candidates(
    image: np.ndarray,
    cfg: TransformConfig,
    max_candidates: int = 8,
) -> list[np.ndarray]:
    variants = transformed_variants(image, cfg)
    base_pts = np.argwhere(image > 0)
    if len(base_pts) == 0:
        return variants[:max_candidates]
    cx0, cy0 = base_pts.mean(axis=0)
    ranked: list[tuple[float, np.ndarray]] = []
    for v in variants:
        pts = np.argwhere(v > 0)
        if len(pts) == 0:
            ranked.append((1e9, v))
            continue
        cx, cy = pts.mean(axis=0)
        d = abs(cx - cx0) + abs(cy - cy0)
        ranked.append((float(d), v))
    ranked.sort(key=lambda t: t[0])
    return [v for _, v in ranked[:max_candidates]]


def best_transformed_score(
    image: np.ndarray,
    value_matrix: np.ndarray,
    penalty_matrix: np.ndarray | None,
    lambda_penalty: float,
    score_mode: str,
    cfg: TransformConfig,
) -> float:
    best = -float("inf")
    for v in transformed_variants(image, cfg):
        s = score_image(v, value_matrix, penalty_matrix, lambda_penalty, score_mode)
        if s > best:
            best = s
    return best


# -----------------------------
# Model build, prediction, calibration
# -----------------------------

@dataclass
class CVMModel:
    fields_pixel: dict[str, np.ndarray]
    fields_skeleton: dict[str, np.ndarray]
    penalties: dict[str, np.ndarray]
    topo_prototypes: dict[str, np.ndarray]
    gammas: dict[str, float]
    gamma_maps: dict[str, np.ndarray]
    distance_kind: str


def build_topology_prototypes(dataset: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for label, samples in dataset.items():
        sig = np.stack([topology_signature(s) for s in samples], axis=0)
        out[label] = sig.mean(axis=0)
    return out


def topology_score(image: np.ndarray, proto: np.ndarray) -> float:
    d = float(np.sum(np.abs(topology_signature(image) - proto)))
    return 1.0 / (1.0 + d)


def train_model(
    dataset: dict[str, list[np.ndarray]],
    shape: tuple[int, int],
    distance_kind: str = "manhattan",
    use_gamma_local: bool = True,
    gamma_by_class: dict[str, float] | None = None,
) -> CVMModel:
    processed: dict[str, list[np.ndarray]] = {}
    for label, samples in dataset.items():
        processed[label] = [preprocess_image(s, output_shape=shape) for s in samples]

    fields_pixel: dict[str, np.ndarray] = {}
    fields_skeleton: dict[str, np.ndarray] = {}
    penalties: dict[str, np.ndarray] = {}
    gammas: dict[str, float] = {}
    gamma_maps: dict[str, np.ndarray] = {}

    for label, samples in processed.items():
        freq = compute_frequency_matrix(samples)
        gamma_l = gamma_by_class[label] if gamma_by_class and label in gamma_by_class else 0.9
        gammas[label] = gamma_l

        gmap = learn_local_gamma_map(samples, base_gamma=max(0.5, gamma_l - 0.05)) if use_gamma_local else None
        gamma_maps[label] = gmap if gmap is not None else np.full(shape, gamma_l, dtype=float)

        field = diffuse_value_matrix(
            freq,
            gamma=gamma_l,
            distance_kind=distance_kind,
            gamma_map=gmap,
        )
        fields_pixel[label] = field
        penalties[label] = build_penalty_matrix(freq)

        skel_samples = [zhang_suen_thinning(s) for s in samples]
        skel_freq = compute_frequency_matrix(skel_samples)
        mean_skel = (skel_freq > 0.5).astype(np.uint8)
        if distance_kind == "geodesic":
            skel_field = diffuse_value_matrix(
                skel_freq,
                gamma=gamma_l,
                distance_kind="geodesic",
                gamma_map=gmap,
                geodesic_skeleton=mean_skel,
            )
        else:
            skel_field = diffuse_value_matrix(
                skel_freq,
                gamma=gamma_l,
                distance_kind=distance_kind,
                gamma_map=gmap,
            )
        fields_skeleton[label] = skel_field

    topo = build_topology_prototypes(processed)
    return CVMModel(
        fields_pixel=fields_pixel,
        fields_skeleton=fields_skeleton,
        penalties=penalties,
        topo_prototypes=topo,
        gammas=gammas,
        gamma_maps=gamma_maps,
        distance_kind=distance_kind,
    )


def predict(
    image: np.ndarray,
    model: CVMModel,
    cfg: TransformConfig,
    lambda_penalty: float = 0.1,
    alpha_pixel: float = 1.0,
    beta_skeleton: float = 0.25,
    beta_topology: float = 0.05,
    score_mode: str = "cosine",
    max_transform_candidates: int | None = None,
) -> tuple[str, dict[str, float]]:
    img = preprocess_image(image, output_shape=next(iter(model.fields_pixel.values())).shape)
    skel = zhang_suen_thinning(img)

    scores: dict[str, float] = {}
    if max_transform_candidates is None:
        pixel_variants = transformed_variants(img, cfg)
        skel_variants = transformed_variants(skel, cfg)
    else:
        pixel_variants = select_transform_candidates(img, cfg, max_candidates=max_transform_candidates)
        skel_variants = select_transform_candidates(skel, cfg, max_candidates=max_transform_candidates)

    def _best(
        variants: list[np.ndarray],
        value_matrix: np.ndarray,
        penalty_matrix: np.ndarray | None,
        lam: float,
    ) -> float:
        best = -float("inf")
        for v in variants:
            s = score_image(v, value_matrix, penalty_matrix, lam, score_mode)
            if s > best:
                best = s
        return best
    for label in model.fields_pixel:
        s_pixel = _best(
            pixel_variants,
            model.fields_pixel[label],
            model.penalties[label],
            lambda_penalty,
        )
        s_skel = _best(
            skel_variants,
            model.fields_skeleton[label],
            None,
            0.0,
        )
        s_topo = topology_score(img, model.topo_prototypes[label])
        scores[label] = alpha_pixel * s_pixel + beta_skeleton * s_skel + beta_topology * s_topo
    pred = max(scores, key=scores.get)
    return pred, scores


def evaluate(
    test_set: list[tuple[str, np.ndarray]],
    model: CVMModel,
    cfg: TransformConfig,
    lambda_penalty: float,
    alpha_pixel: float,
    beta_skeleton: float,
    beta_topology: float,
    score_mode: str = "cosine",
    max_transform_candidates: int | None = None,
) -> tuple[float, list[str], np.ndarray]:
    labels = sorted(model.fields_pixel.keys())
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    ok = 0
    for truth, img in test_set:
        pred, _ = predict(
            img,
            model,
            cfg,
            lambda_penalty=lambda_penalty,
            alpha_pixel=alpha_pixel,
            beta_skeleton=beta_skeleton,
            beta_topology=beta_topology,
            score_mode=score_mode,
            max_transform_candidates=max_transform_candidates,
        )
        cm[idx[truth], idx[pred]] += 1
        if pred == truth:
            ok += 1
    return ok / len(test_set), labels, cm


def tune_hyperparams(
    train_set: dict[str, list[np.ndarray]],
    val_set: list[tuple[str, np.ndarray]],
    shape: tuple[int, int],
) -> tuple[CVMModel, dict[str, float], TransformConfig, float]:
    best_acc = -1.0
    best_model = None
    best_cfg = None
    best_params = None

    dist_kinds = ["manhattan"]
    gamma_vals = [0.85, 0.95]
    lambda_vals = [0.05]
    beta_topo_vals = [0.0, 0.05]
    beta_skel_vals = [0.15]

    for dist in dist_kinds:
        for gamma_global in gamma_vals:
            gamma_by_class = {k: gamma_global for k in train_set.keys()}
            model = train_model(
                train_set,
                shape,
                distance_kind=dist,
                use_gamma_local=True,
                gamma_by_class=gamma_by_class,
            )
            cfg = TransformConfig(
                translations=[(0, 0), (1, 0)],
                rotations_deg=[0],
                scales=[1.0],
                shears=[0.0],
            )
            for lam in lambda_vals:
                for btop in beta_topo_vals:
                    for bsk in beta_skel_vals:
                        acc, _, _ = evaluate(
                            val_set,
                            model,
                            cfg,
                            lambda_penalty=lam,
                            alpha_pixel=1.0,
                            beta_skeleton=bsk,
                            beta_topology=btop,
                            score_mode="cosine",
                        )
                        if acc > best_acc:
                            best_acc = acc
                            best_model = model
                            best_cfg = cfg
                            best_params = {
                                "gamma_global": gamma_global,
                                "lambda_penalty": lam,
                                "beta_topology": btop,
                                "beta_skeleton": bsk,
                                "distance_kind": dist,
                            }
    assert best_model is not None and best_cfg is not None and best_params is not None
    return best_model, best_params, best_cfg, best_acc


# -----------------------------
# Toy data + full demo
# -----------------------------

def _demo_dataset() -> dict[str, list[np.ndarray]]:
    A1 = np.array([
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
    ])
    A2 = np.array([
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
    ])
    H1 = np.array([
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
    ])
    H2 = np.array([
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
    ])
    O1 = np.array([
        [0, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1, 1, 0],
    ])
    O2 = np.array([
        [0, 1, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0, 0],
    ])
    C1 = np.array([
        [0, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1, 1, 0],
    ])
    C2 = np.array([
        [0, 1, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0, 0],
    ])
    return {"A": [A1, A2], "H": [H1, H2], "O": [O1, O2], "C": [C1, C2]}


def split_toy_dataset(dataset: dict[str, list[np.ndarray]]) -> tuple[dict[str, list[np.ndarray]], list[tuple[str, np.ndarray]]]:
    train = {k: [v[0]] for k, v in dataset.items()}
    val = [(k, v[1]) for k, v in dataset.items()]
    # Add stress cases.
    val.append(("A", np.roll(dataset["A"][1], 1, axis=1)))
    val.append(("O", np.roll(dataset["O"][1], -1, axis=0)))
    return train, val


def _add_salt_pepper(image: np.ndarray, n_flips: int = 2) -> np.ndarray:
    out = image.copy().astype(np.uint8)
    h, w = out.shape
    for k in range(n_flips):
        x = (k * 3 + 1) % h
        y = (k * 5 + 2) % w
        out[x, y] = 1 - out[x, y]
    return out


def _break_stroke(image: np.ndarray, n_zero: int = 2) -> np.ndarray:
    out = image.copy().astype(np.uint8)
    pts = np.argwhere(out > 0)
    for i in range(min(n_zero, len(pts))):
        x, y = pts[(i * 7) % len(pts)]
        out[x, y] = 0
    return out


def build_stress_val_set(dataset: dict[str, list[np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    stress: list[tuple[str, np.ndarray]] = []
    cfg = TransformConfig(
        translations=[(0, 0), (1, 0), (0, 1)],
        rotations_deg=[-12, 0, 12],
        scales=[0.9, 1.0, 1.1],
        shears=[-0.08, 0.0, 0.08],
    )
    for label, samples in dataset.items():
        src = samples[1]
        variants = transformed_variants(src, cfg)
        picks = [variants[0], variants[len(variants) // 2], variants[-1]]
        for v in picks:
            stress.append((label, _add_salt_pepper(v, n_flips=2)))
            stress.append((label, _break_stroke(v, n_zero=2)))
    return stress


def kfold_split_indices(n: int, k: int) -> list[list[int]]:
    folds: list[list[int]] = [[] for _ in range(k)]
    for i in range(n):
        folds[i % k].append(i)
    return folds


def evaluate_cvm_on_dataset(
    train_set: dict[str, list[np.ndarray]],
    val_set: list[tuple[str, np.ndarray]],
    shape: tuple[int, int] = (24, 24),
) -> tuple[float, dict[str, float]]:
    model, params, cfg, _ = tune_hyperparams(train_set, val_set, shape=shape)
    acc, _, _ = evaluate(
        val_set,
        model,
        cfg,
        lambda_penalty=params["lambda_penalty"],
        alpha_pixel=1.0,
        beta_skeleton=params["beta_skeleton"],
        beta_topology=params["beta_topology"],
        score_mode="cosine",
    )
    return acc, params


def cross_validate_toy(dataset: dict[str, list[np.ndarray]], k: int = 2) -> tuple[float, list[float]]:
    labels = sorted(dataset.keys())
    n = min(len(dataset[l]) for l in labels)
    folds = kfold_split_indices(n, k)
    scores: list[float] = []
    for fold_id in range(k):
        val_idx = set(folds[fold_id])
        train_set: dict[str, list[np.ndarray]] = {}
        val_set: list[tuple[str, np.ndarray]] = []
        for label in labels:
            train_set[label] = [img for i, img in enumerate(dataset[label][:n]) if i not in val_idx]
            val_set.extend((label, img) for i, img in enumerate(dataset[label][:n]) if i in val_idx)
        if any(len(v) == 0 for v in train_set.values()):
            continue
        acc, _ = evaluate_cvm_on_dataset(train_set, val_set, shape=(24, 24))
        scores.append(acc)
    mean_acc = float(np.mean(scores)) if scores else 0.0
    return mean_acc, scores


def benchmark_sklearn_digits(max_train_per_class: int = 80, max_test_per_class: int = 30) -> tuple[float, int] | None:
    try:
        from sklearn.datasets import load_digits
    except Exception:
        return None

    digits = load_digits()
    X = digits.images.astype(float)
    y = digits.target.astype(int)
    classes = sorted(set(int(v) for v in y))

    train_set: dict[str, list[np.ndarray]] = {str(c): [] for c in classes}
    test_set: list[tuple[str, np.ndarray]] = []
    train_count = {c: 0 for c in classes}
    test_count = {c: 0 for c in classes}

    for img, label in zip(X, y):
        c = int(label)
        if train_count[c] < max_train_per_class:
            train_set[str(c)].append((img / 16.0 > 0.35).astype(np.uint8))
            train_count[c] += 1
        elif test_count[c] < max_test_per_class:
            test_set.append((str(c), (img / 16.0 > 0.35).astype(np.uint8)))
            test_count[c] += 1
        if all(train_count[c] >= max_train_per_class and test_count[c] >= max_test_per_class for c in classes):
            break

    model = train_model(
        train_set,
        shape=(24, 24),
        distance_kind="manhattan",
        use_gamma_local=True,
        gamma_by_class={str(c): 0.9 for c in classes},
    )
    cfg = TransformConfig(
        translations=[(0, 0), (1, 0), (0, 1)],
        rotations_deg=[0],
        scales=[1.0],
        shears=[0.0],
    )
    acc, _, _ = evaluate(
        test_set,
        model,
        cfg,
        lambda_penalty=0.05,
        alpha_pixel=1.0,
        beta_skeleton=0.15,
        beta_topology=0.0,
        score_mode="cosine",
    )
    return acc, len(test_set)


def _build_digits_split(
    max_train_per_class: int,
    max_test_per_class: int,
) -> tuple[dict[str, list[np.ndarray]], list[tuple[str, np.ndarray]]] | None:
    try:
        from sklearn.datasets import load_digits
    except Exception:
        return None
    digits = load_digits()
    X = digits.images.astype(float)
    y = digits.target.astype(int)
    classes = sorted(set(int(v) for v in y))

    train_set: dict[str, list[np.ndarray]] = {str(c): [] for c in classes}
    test_set: list[tuple[str, np.ndarray]] = []
    train_count = {c: 0 for c in classes}
    test_count = {c: 0 for c in classes}

    for img, label in zip(X, y):
        c = int(label)
        img_bin = (img / 16.0 > 0.35).astype(np.uint8)
        if train_count[c] < max_train_per_class:
            train_set[str(c)].append(img_bin)
            train_count[c] += 1
        elif test_count[c] < max_test_per_class:
            test_set.append((str(c), img_bin))
            test_count[c] += 1
        if all(train_count[c] >= max_train_per_class and test_count[c] >= max_test_per_class for c in classes):
            break
    return train_set, test_set


def gamma_study_digits(
    gamma_values: list[float] | None = None,
    max_train_per_class: int = 80,
    max_test_per_class: int = 30,
) -> list[tuple[float, float]]:
    """Run a disjoint train/test gamma sweep on sklearn digits."""
    if gamma_values is None:
        gamma_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    split = _build_digits_split(max_train_per_class, max_test_per_class)
    if split is None:
        return []
    train_set, test_set = split
    cfg = TransformConfig(
        translations=[(0, 0), (1, 0), (0, 1)],
        rotations_deg=[0],
        scales=[1.0],
        shears=[0.0],
    )
    results: list[tuple[float, float]] = []
    for gamma in gamma_values:
        model = train_model(
            train_set,
            shape=(24, 24),
            distance_kind="manhattan",
            use_gamma_local=False,
            gamma_by_class={k: gamma for k in train_set.keys()},
        )
        acc, _, _ = evaluate(
            test_set,
            model,
            cfg,
            lambda_penalty=0.05,
            alpha_pixel=1.0,
            beta_skeleton=0.15,
            beta_topology=0.0,
            score_mode="cosine",
        )
        results.append((gamma, acc))
    return results


def gamma_study_digits_three_way(
    gamma_values: list[float] | None = None,
    max_train_per_class: int = 60,
    max_val_per_class: int = 20,
    max_test_per_class: int = 20,
) -> dict[str, object] | None:
    """Train/val/test study + explicit no-diffusion point."""
    try:
        from sklearn.datasets import load_digits
    except Exception:
        return None
    if gamma_values is None:
        gamma_values = [0.1, 0.2, 0.3, 0.4, 0.5]

    digits = load_digits()
    X = digits.images.astype(float)
    y = digits.target.astype(int)
    classes = sorted(set(int(v) for v in y))

    train_set: dict[str, list[np.ndarray]] = {str(c): [] for c in classes}
    val_set: list[tuple[str, np.ndarray]] = []
    test_set: list[tuple[str, np.ndarray]] = []
    c_train = {c: 0 for c in classes}
    c_val = {c: 0 for c in classes}
    c_test = {c: 0 for c in classes}

    for img, label in zip(X, y):
        c = int(label)
        img_bin = (img / 16.0 > 0.35).astype(np.uint8)
        if c_train[c] < max_train_per_class:
            train_set[str(c)].append(img_bin)
            c_train[c] += 1
        elif c_val[c] < max_val_per_class:
            val_set.append((str(c), img_bin))
            c_val[c] += 1
        elif c_test[c] < max_test_per_class:
            test_set.append((str(c), img_bin))
            c_test[c] += 1
        if all(
            c_train[c] >= max_train_per_class and c_val[c] >= max_val_per_class and c_test[c] >= max_test_per_class
            for c in classes
        ):
            break

    cfg = TransformConfig(
        translations=[(0, 0), (1, 0), (0, 1)],
        rotations_deg=[0],
        scales=[1.0],
        shears=[0.0],
    )
    rows: list[tuple[str, float, float]] = []

    # Diffusion points.
    for g in gamma_values:
        model = train_model(
            train_set,
            shape=(24, 24),
            distance_kind="manhattan",
            use_gamma_local=False,
            gamma_by_class={k: g for k in train_set.keys()},
        )
        val_acc, _, _ = evaluate(
            val_set,
            model,
            cfg,
            lambda_penalty=0.05,
            alpha_pixel=1.0,
            beta_skeleton=0.15,
            beta_topology=0.0,
            score_mode="cosine",
        )
        test_acc, _, _ = evaluate(
            test_set,
            model,
            cfg,
            lambda_penalty=0.05,
            alpha_pixel=1.0,
            beta_skeleton=0.15,
            beta_topology=0.0,
            score_mode="cosine",
        )
        rows.append((f"gamma={g:.2f}", val_acc, test_acc))

    # No-diffusion point: V_L = F_L.
    model_nd = train_model(
        train_set,
        shape=(24, 24),
        distance_kind="manhattan",
        use_gamma_local=False,
        gamma_by_class={k: 0.5 for k in train_set.keys()},
    )
    # Override pixel field with direct frequency map F_L (preprocessed average).
    for label, samples in train_set.items():
        proc = [preprocess_image(s, output_shape=(24, 24)) for s in samples]
        freq = compute_frequency_matrix(proc)
        model_nd.fields_pixel[label] = freq
        model_nd.fields_skeleton[label] = compute_frequency_matrix([zhang_suen_thinning(p) for p in proc])
    nd_val, _, _ = evaluate(
        val_set,
        model_nd,
        cfg,
        lambda_penalty=0.05,
        alpha_pixel=1.0,
        beta_skeleton=0.15,
        beta_topology=0.0,
        score_mode="cosine",
    )
    nd_test, _, _ = evaluate(
        test_set,
        model_nd,
        cfg,
        lambda_penalty=0.05,
        alpha_pixel=1.0,
        beta_skeleton=0.15,
        beta_topology=0.0,
        score_mode="cosine",
    )
    rows.append(("no-diffusion", nd_val, nd_test))

    best = max(rows, key=lambda r: r[1])
    return {
        "rows": rows,
        "best_on_val": best[0],
        "best_val_acc": best[1],
        "test_acc_of_best": best[2],
        "n_val": len(val_set),
        "n_test": len(test_set),
    }


def _shift_no_recenter(image: np.ndarray, dx: int, dy: int, shape: tuple[int, int] = (24, 24)) -> np.ndarray:
    """Embed in larger canvas with translation, no recentering."""
    src = image.astype(np.uint8)
    out = np.zeros(shape, dtype=np.uint8)
    h, w = src.shape
    x0 = max(0, 8 + dx)
    y0 = max(0, 8 + dy)
    x1 = min(shape[0], x0 + h)
    y1 = min(shape[1], y0 + w)
    sx = 0
    sy = 0
    out[x0:x1, y0:y1] = src[sx : sx + (x1 - x0), sy : sy + (y1 - y0)]
    return out


def gamma_study_digits_shifted(
    gamma_values: list[float] | None = None,
    max_train_per_class: int = 60,
    max_val_per_class: int = 20,
    max_test_per_class: int = 20,
    shift_range: int = 3,
) -> dict[str, object] | None:
    """Train/val/test on shifted digits without recentering to test spatial variability."""
    try:
        from sklearn.datasets import load_digits
    except Exception:
        return None
    if gamma_values is None:
        gamma_values = [0.1, 0.2, 0.3, 0.4, 0.5]

    digits = load_digits()
    X = digits.images.astype(float)
    y = digits.target.astype(int)
    classes = sorted(set(int(v) for v in y))

    train_set: dict[str, list[np.ndarray]] = {str(c): [] for c in classes}
    val_set: list[tuple[str, np.ndarray]] = []
    test_set: list[tuple[str, np.ndarray]] = []
    c_train = {c: 0 for c in classes}
    c_val = {c: 0 for c in classes}
    c_test = {c: 0 for c in classes}

    for idx, (img, label) in enumerate(zip(X, y)):
        c = int(label)
        img_bin = (img / 16.0 > 0.35).astype(np.uint8)
        # Deterministic pseudo-random shifts tied to index.
        dx = (idx % (2 * shift_range + 1)) - shift_range
        dy = ((idx * 3) % (2 * shift_range + 1)) - shift_range
        shifted = _shift_no_recenter(img_bin, dx, dy, shape=(24, 24))
        if c_train[c] < max_train_per_class:
            train_set[str(c)].append(shifted)
            c_train[c] += 1
        elif c_val[c] < max_val_per_class:
            val_set.append((str(c), shifted))
            c_val[c] += 1
        elif c_test[c] < max_test_per_class:
            test_set.append((str(c), shifted))
            c_test[c] += 1
        if all(
            c_train[c] >= max_train_per_class and c_val[c] >= max_val_per_class and c_test[c] >= max_test_per_class
            for c in classes
        ):
            break

    cfg = TransformConfig(
        translations=[(0, 0)],
        rotations_deg=[0],
        scales=[1.0],
        shears=[0.0],
    )
    rows: list[tuple[str, float, float]] = []

    # Build model directly from already shifted data (disable preprocess recenter effects).
    def train_shifted_model(gamma: float, no_diffusion: bool = False) -> CVMModel:
        fields_pixel: dict[str, np.ndarray] = {}
        fields_skeleton: dict[str, np.ndarray] = {}
        penalties: dict[str, np.ndarray] = {}
        topo: dict[str, np.ndarray] = {}
        for label, samples in train_set.items():
            freq = compute_frequency_matrix(samples)
            if no_diffusion:
                fields_pixel[label] = freq
            else:
                fields_pixel[label] = diffuse_value_matrix(freq, gamma=gamma, distance_kind="manhattan", gamma_map=None)
            sk = [zhang_suen_thinning(s) for s in samples]
            skf = compute_frequency_matrix(sk)
            fields_skeleton[label] = skf if no_diffusion else diffuse_value_matrix(skf, gamma=gamma, distance_kind="manhattan", gamma_map=None)
            penalties[label] = build_penalty_matrix(freq)
            topo[label] = np.stack([topology_signature(s) for s in samples], axis=0).mean(axis=0)
        return CVMModel(
            fields_pixel=fields_pixel,
            fields_skeleton=fields_skeleton,
            penalties=penalties,
            topo_prototypes=topo,
            gammas={k: gamma for k in train_set.keys()},
            gamma_maps={k: np.full((24, 24), gamma, dtype=float) for k in train_set.keys()},
            distance_kind="manhattan",
        )

    def eval_shifted_set(model: CVMModel, eval_set: list[tuple[str, np.ndarray]]) -> float:
        labels = list(model.fields_pixel.keys())
        correct = 0
        for truth, img in eval_set:
            imgf = img.astype(float)
            sk = zhang_suen_thinning(img)
            best_label = None
            best_score = -float("inf")
            for label in labels:
                s1 = score_image(imgf, model.fields_pixel[label], model.penalties[label], 0.05, "cosine")
                s2 = score_image(sk.astype(float), model.fields_skeleton[label], None, 0.0, "cosine")
                score = s1 + 0.15 * s2
                if score > best_score:
                    best_score = score
                    best_label = label
            if best_label == truth:
                correct += 1
        return correct / len(eval_set) if eval_set else 0.0

    for g in gamma_values:
        model = train_shifted_model(gamma=g, no_diffusion=False)
        val_acc = eval_shifted_set(model, val_set)
        test_acc = eval_shifted_set(model, test_set)
        rows.append((f"gamma={g:.2f}", val_acc, test_acc))

    model_nd = train_shifted_model(gamma=0.5, no_diffusion=True)
    nd_val = eval_shifted_set(model_nd, val_set)
    nd_test = eval_shifted_set(model_nd, test_set)
    rows.append(("no-diffusion", nd_val, nd_test))

    best = max(rows, key=lambda r: r[1])
    return {
        "rows": rows,
        "best_on_val": best[0],
        "best_val_acc": best[1],
        "test_acc_of_best": best[2],
        "n_val": len(val_set),
        "n_test": len(test_set),
    }


def gamma_study_digits_shifted_small(
    gamma_values: list[float] | None = None,
    max_train_per_class: int = 60,
    max_val_per_class: int = 20,
    max_test_per_class: int = 20,
    shift_values: tuple[int, ...] = (-2, -1, 0, 1, 2),
) -> dict[str, object] | None:
    """Fair diffusion test: small shifts, no recentering, no explicit translations."""
    try:
        from sklearn.datasets import load_digits
    except Exception:
        return None
    if gamma_values is None:
        gamma_values = [0.3, 0.5, 0.7]

    digits = load_digits()
    X = digits.images.astype(float)
    y = digits.target.astype(int)
    classes = sorted(set(int(v) for v in y))

    train_set: dict[str, list[np.ndarray]] = {str(c): [] for c in classes}
    val_set: list[tuple[str, np.ndarray]] = []
    test_set: list[tuple[str, np.ndarray]] = []
    c_train = {c: 0 for c in classes}
    c_val = {c: 0 for c in classes}
    c_test = {c: 0 for c in classes}

    for idx, (img, label) in enumerate(zip(X, y)):
        c = int(label)
        img_bin = (img / 16.0 > 0.35).astype(np.uint8)
        dx = shift_values[idx % len(shift_values)]
        dy = shift_values[(idx * 3) % len(shift_values)]
        shifted = _shift_no_recenter(img_bin, dx, dy, shape=(24, 24))
        if c_train[c] < max_train_per_class:
            train_set[str(c)].append(shifted)
            c_train[c] += 1
        elif c_val[c] < max_val_per_class:
            val_set.append((str(c), shifted))
            c_val[c] += 1
        elif c_test[c] < max_test_per_class:
            test_set.append((str(c), shifted))
            c_test[c] += 1
        if all(
            c_train[c] >= max_train_per_class and c_val[c] >= max_val_per_class and c_test[c] >= max_test_per_class
            for c in classes
        ):
            break

    def train_shifted_model(gamma: float, no_diffusion: bool = False) -> CVMModel:
        fields_pixel: dict[str, np.ndarray] = {}
        fields_skeleton: dict[str, np.ndarray] = {}
        penalties: dict[str, np.ndarray] = {}
        topo: dict[str, np.ndarray] = {}
        for label, samples in train_set.items():
            freq = compute_frequency_matrix(samples)
            fields_pixel[label] = freq if no_diffusion else diffuse_value_matrix(freq, gamma=gamma, distance_kind="manhattan")
            sk = [zhang_suen_thinning(s) for s in samples]
            skf = compute_frequency_matrix(sk)
            fields_skeleton[label] = skf if no_diffusion else diffuse_value_matrix(skf, gamma=gamma, distance_kind="manhattan")
            penalties[label] = build_penalty_matrix(freq)
            topo[label] = np.stack([topology_signature(s) for s in samples], axis=0).mean(axis=0)
        return CVMModel(
            fields_pixel=fields_pixel,
            fields_skeleton=fields_skeleton,
            penalties=penalties,
            topo_prototypes=topo,
            gammas={k: gamma for k in train_set.keys()},
            gamma_maps={k: np.full((24, 24), gamma, dtype=float) for k in train_set.keys()},
            distance_kind="manhattan",
        )

    def eval_shifted_set(model: CVMModel, eval_set: list[tuple[str, np.ndarray]]) -> float:
        labels = list(model.fields_pixel.keys())
        correct = 0
        for truth, img in eval_set:
            imgf = img.astype(float)
            sk = zhang_suen_thinning(img)
            best_label = None
            best_score = -float("inf")
            for label in labels:
                s1 = score_image(imgf, model.fields_pixel[label], model.penalties[label], 0.05, "cosine")
                s2 = score_image(sk.astype(float), model.fields_skeleton[label], None, 0.0, "cosine")
                score = s1 + 0.15 * s2
                if score > best_score:
                    best_score = score
                    best_label = label
            if best_label == truth:
                correct += 1
        return correct / len(eval_set) if eval_set else 0.0

    rows: list[tuple[str, float, float]] = []
    for g in gamma_values:
        model = train_shifted_model(gamma=g, no_diffusion=False)
        rows.append((f"gamma={g:.2f}", eval_shifted_set(model, val_set), eval_shifted_set(model, test_set)))
    model_nd = train_shifted_model(gamma=0.5, no_diffusion=True)
    rows.append(("no-diffusion", eval_shifted_set(model_nd, val_set), eval_shifted_set(model_nd, test_set)))

    best = max(rows, key=lambda r: r[1])
    chance = 1.0 / len(classes)
    feasible = max(r[2] for r in rows) > max(0.5, chance + 0.05)
    return {
        "rows": rows,
        "best_on_val": best[0],
        "best_val_acc": best[1],
        "test_acc_of_best": best[2],
        "n_val": len(val_set),
        "n_test": len(test_set),
        "chance": chance,
        "feasible_above_chance": feasible,
    }


def benchmark_sklearn_digits_classic(
    max_train_per_class: int = 80,
    max_test_per_class: int = 30,
) -> dict[str, float] | None:
    try:
        from sklearn.datasets import load_digits
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
    except Exception:
        return None

    digits = load_digits()
    X = digits.images.astype(float)
    y = digits.target.astype(int)
    classes = sorted(set(int(v) for v in y))

    X_train: list[np.ndarray] = []
    y_train: list[int] = []
    X_test: list[np.ndarray] = []
    y_test: list[int] = []
    train_count = {c: 0 for c in classes}
    test_count = {c: 0 for c in classes}

    for img, label in zip(X, y):
        c = int(label)
        flat = (img / 16.0).reshape(-1)
        if train_count[c] < max_train_per_class:
            X_train.append(flat)
            y_train.append(c)
            train_count[c] += 1
        elif test_count[c] < max_test_per_class:
            X_test.append(flat)
            y_test.append(c)
            test_count[c] += 1
        if all(train_count[c] >= max_train_per_class and test_count[c] >= max_test_per_class for c in classes):
            break

    Xtr = np.array(X_train, dtype=float)
    Xte = np.array(X_test, dtype=float)
    ytr = np.array(y_train, dtype=int)
    yte = np.array(y_test, dtype=int)

    t0 = time.perf_counter()
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(Xtr, ytr)
    knn_acc = float(np.mean(knn.predict(Xte) == yte))
    knn_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(Xtr, ytr)
    lr_acc = float(np.mean(lr.predict(Xte) == yte))
    lr_ms = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    svm = SVC(kernel="rbf", gamma="scale", C=5.0)
    svm.fit(Xtr, ytr)
    svm_acc = float(np.mean(svm.predict(Xte) == yte))
    svm_ms = (time.perf_counter() - t2) * 1000.0

    return {
        "n_test": float(len(yte)),
        "knn_acc": knn_acc,
        "knn_ms": knn_ms,
        "logreg_acc": lr_acc,
        "logreg_ms": lr_ms,
        "svm_acc": svm_acc,
        "svm_ms": svm_ms,
    }


def ablation_cvm(
    train_set: dict[str, list[np.ndarray]],
    val_set: list[tuple[str, np.ndarray]],
    max_transform_candidates: int | None = None,
) -> dict[str, float]:
    model = train_model(
        train_set,
        shape=(24, 24),
        distance_kind="manhattan",
        use_gamma_local=True,
        gamma_by_class={k: 0.85 for k in train_set.keys()},
    )
    cfg_none = TransformConfig(translations=[(0, 0)], rotations_deg=[0], scales=[1.0], shears=[0.0])
    cfg_trans = TransformConfig(
        translations=[(0, 0), (1, 0)],
        rotations_deg=[0],
        scales=[1.0],
        shears=[0.0],
    )

    a0, _, _ = evaluate(
        val_set,
        model,
        cfg_none,
        lambda_penalty=0.0,
        alpha_pixel=1.0,
        beta_skeleton=0.0,
        beta_topology=0.0,
        score_mode="cosine",
        max_transform_candidates=max_transform_candidates,
    )
    a1, _, _ = evaluate(
        val_set,
        model,
        cfg_none,
        lambda_penalty=0.05,
        alpha_pixel=1.0,
        beta_skeleton=0.0,
        beta_topology=0.0,
        score_mode="cosine",
        max_transform_candidates=max_transform_candidates,
    )
    a2, _, _ = evaluate(
        val_set,
        model,
        cfg_trans,
        lambda_penalty=0.05,
        alpha_pixel=1.0,
        beta_skeleton=0.15,
        beta_topology=0.0,
        score_mode="cosine",
        max_transform_candidates=max_transform_candidates,
    )
    a3, _, _ = evaluate(
        val_set,
        model,
        cfg_trans,
        lambda_penalty=0.05,
        alpha_pixel=1.0,
        beta_skeleton=0.15,
        beta_topology=0.05,
        score_mode="cosine",
        max_transform_candidates=max_transform_candidates,
    )
    return {
        "champ_seul": a0,
        "champ_plus_penalite": a1,
        "champ_penalite_transfos": a2,
        "champ_penalite_transfos_topo": a3,
    }


def template_baseline_predict(
    image: np.ndarray,
    templates: dict[str, np.ndarray],
    centered: bool = False,
) -> str:
    best_label = ""
    best_score = -float("inf")
    for label, template in templates.items():
        x = preprocess_image(image, template.shape) if centered else image.astype(float)
        t = template.astype(float)
        denom = float(np.linalg.norm(x) * np.linalg.norm(t))
        score = 0.0 if denom <= 0 else float(np.sum(x * t)) / denom
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


def evaluate_template_baseline(
    test_set: list[tuple[str, np.ndarray]],
    templates: dict[str, np.ndarray],
    centered: bool = False,
) -> float:
    correct = 0
    for truth, image in test_set:
        pred = template_baseline_predict(image, templates, centered=centered)
        if pred == truth:
            correct += 1
    return correct / len(test_set) if test_set else 0.0


def distance_transform_to_active(binary: np.ndarray) -> np.ndarray:
    """Manhattan distance to nearest active pixel (1)."""
    b = (binary > 0).astype(np.uint8)
    h, w = b.shape
    inf = h + w + 5
    dist = np.full((h, w), inf, dtype=float)
    q = deque()
    for x in range(h):
        for y in range(w):
            if b[x, y] == 1:
                dist[x, y] = 0.0
                q.append((x, y))
    if not q:
        return dist
    while q:
        x, y = q.popleft()
        base = dist[x, y]
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < h and 0 <= ny < w and base + 1 < dist[nx, ny]:
                dist[nx, ny] = base + 1
                q.append((nx, ny))
    return dist


def _mean_distance_on_foreground(dist_map: np.ndarray, binary: np.ndarray) -> float:
    pts = np.argwhere(binary > 0)
    if len(pts) == 0:
        return float(np.max(dist_map))
    return float(np.mean([dist_map[x, y] for x, y in pts]))


def distance_transform_baseline_predict(
    image: np.ndarray,
    templates: dict[str, np.ndarray],
) -> str:
    x = preprocess_image(image, next(iter(templates.values())).shape).astype(np.uint8)
    best_label = ""
    best_score = -float("inf")
    for label, template in templates.items():
        dt = distance_transform_to_active(template)
        mean_d = _mean_distance_on_foreground(dt, x)
        score = -mean_d
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


def chamfer_baseline_predict(
    image: np.ndarray,
    templates: dict[str, np.ndarray],
) -> str:
    x = preprocess_image(image, next(iter(templates.values())).shape).astype(np.uint8)
    dt_x = distance_transform_to_active(x)
    best_label = ""
    best_score = -float("inf")
    for label, template in templates.items():
        t = template.astype(np.uint8)
        dt_t = distance_transform_to_active(t)
        d_xt = _mean_distance_on_foreground(dt_t, x)
        d_tx = _mean_distance_on_foreground(dt_x, t)
        chamfer = 0.5 * (d_xt + d_tx)
        score = -chamfer
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


def evaluate_predictor(
    test_set: list[tuple[str, np.ndarray]],
    predictor,
) -> float:
    correct = 0
    for truth, image in test_set:
        pred = predictor(image)
        if pred == truth:
            correct += 1
    return correct / len(test_set) if test_set else 0.0


def explain_prediction(
    image: np.ndarray,
    truth_label: str,
    model: CVMModel,
    cfg: TransformConfig,
    lambda_penalty: float = 0.05,
    score_mode: str = "cosine",
    max_transform_candidates: int = 8,
    show_plot: bool = True,
) -> tuple[str, dict[str, float]]:
    """Interpretability view for one sample: truth vs predicted class fields."""
    pred, scores = predict(
        image,
        model,
        cfg,
        lambda_penalty=lambda_penalty,
        alpha_pixel=1.0,
        beta_skeleton=0.15,
        beta_topology=0.05,
        score_mode=score_mode,
        max_transform_candidates=max_transform_candidates,
    )
    proc = preprocess_image(image, output_shape=next(iter(model.fields_pixel.values())).shape)
    true_field = model.fields_pixel[truth_label]
    pred_field = model.fields_pixel[pred]
    true_pen = model.penalties[truth_label]
    pred_pen = model.penalties[pred]

    act_true = proc.astype(float) * true_field
    act_pred = proc.astype(float) * pred_field
    pen_true = proc.astype(float) * true_pen
    pen_pred = proc.astype(float) * pred_pen

    if show_plot:
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        axes[0, 0].imshow(proc, cmap="gray_r", interpolation="nearest")
        axes[0, 0].set_title(f"Input (truth={truth_label})")
        axes[0, 1].imshow(true_field, cmap="hot", interpolation="nearest")
        axes[0, 1].set_title(f"Field true={truth_label}")
        axes[0, 2].imshow(act_true, cmap="viridis", interpolation="nearest")
        axes[0, 2].set_title("Activation true (I*V)")
        axes[0, 3].imshow(pen_true, cmap="magma", interpolation="nearest")
        axes[0, 3].set_title("Penalty true (I*P)")

        axes[1, 0].imshow(proc, cmap="gray_r", interpolation="nearest")
        axes[1, 0].set_title(f"Input (pred={pred})")
        axes[1, 1].imshow(pred_field, cmap="hot", interpolation="nearest")
        axes[1, 1].set_title(f"Field pred={pred}")
        axes[1, 2].imshow(act_pred, cmap="viridis", interpolation="nearest")
        axes[1, 2].set_title("Activation pred (I*V)")
        axes[1, 3].imshow(pen_pred, cmap="magma", interpolation="nearest")
        axes[1, 3].set_title("Penalty pred (I*P)")

        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.show()
    return pred, scores


def build_ambiguous_val_set(dataset: dict[str, list[np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    """Focused set on confusing pairs: A/H and C/O."""
    out: list[tuple[str, np.ndarray]] = []
    cfg = TransformConfig(
        translations=[(0, 0), (1, 0)],
        rotations_deg=[-8, 0, 8],
        scales=[0.95, 1.0, 1.05],
        shears=[0.0],
    )
    for label in ("A", "H", "C", "O"):
        src = dataset[label][1]
        variants = transformed_variants(src, cfg)
        out.append((label, variants[len(variants) // 3]))
        out.append((label, variants[(2 * len(variants)) // 3]))
    return out


def benchmark_levels(
    train_set: dict[str, list[np.ndarray]],
    levels: dict[str, list[tuple[str, np.ndarray]]],
) -> None:
    centered_templates = {k: preprocess_image(v[0], (24, 24)).astype(float) for k, v in train_set.items()}
    model = train_model(
        train_set,
        shape=(24, 24),
        distance_kind="manhattan",
        use_gamma_local=True,
        gamma_by_class={k: 0.85 for k in train_set.keys()},
    )
    cfg = TransformConfig(
        translations=[(0, 0), (1, 0), (0, 1)],
        rotations_deg=[-8, 0, 8],
        scales=[0.95, 1.0, 1.05],
        shears=[0.0],
    )
    for level_name, level_set in levels.items():
        acc_cvm, _, _ = evaluate(
            level_set,
            model,
            cfg,
            lambda_penalty=0.05,
            alpha_pixel=1.0,
            beta_skeleton=0.15,
            beta_topology=0.05,
            score_mode="cosine",
            max_transform_candidates=8,
        )
        acc_tpl = evaluate_template_baseline(level_set, centered_templates, centered=True)
        acc_chamfer = evaluate_predictor(
            level_set,
            lambda img: chamfer_baseline_predict(img, centered_templates),
        )
        print(
            f"Level {level_name}: CVM={acc_cvm:.3f}, "
            f"TemplateCentered={acc_tpl:.3f}, Chamfer={acc_chamfer:.3f}"
        )


def run_demo(show_plot: bool = True) -> None:
    dataset = _demo_dataset()
    train_set, val_set = split_toy_dataset(dataset)

    base_model = train_model(train_set, shape=(24, 24), distance_kind="manhattan", use_gamma_local=False)
    base_cfg = TransformConfig(translations=[(0, 0)], rotations_deg=[0], scales=[1.0], shears=[0.0])

    t0 = time.perf_counter()
    base_acc, labels, base_cm = evaluate(
        val_set,
        base_model,
        base_cfg,
        lambda_penalty=0.0,
        alpha_pixel=1.0,
        beta_skeleton=0.0,
        beta_topology=0.0,
    )
    base_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    tuned_model, tuned_params, tuned_cfg, tuned_val_acc = tune_hyperparams(train_set, val_set, shape=(24, 24))
    tuning_ms = (time.perf_counter() - t1) * 1000.0
    t1b = time.perf_counter()
    enh_acc, _, enh_cm = evaluate(
        val_set,
        tuned_model,
        tuned_cfg,
        lambda_penalty=tuned_params["lambda_penalty"],
        alpha_pixel=1.0,
        beta_skeleton=tuned_params["beta_skeleton"],
        beta_topology=tuned_params["beta_topology"],
    )
    enh_infer_ms = (time.perf_counter() - t1b) * 1000.0

    print(f"Base accuracy: {base_acc:.3f}")
    print(f"Base eval time (ms): {base_ms:.1f}")
    print("Base confusion (rows=true, cols=pred):")
    print(base_cm)
    print("Labels:", labels)

    print(f"Enhanced tuned accuracy: {enh_acc:.3f}")
    print(f"Enhanced tuning time (ms): {tuning_ms:.1f}")
    print(f"Enhanced inference time (ms): {enh_infer_ms:.1f}")
    print("Enhanced confusion (rows=true, cols=pred):")
    print(enh_cm)
    print("Best params:", tuned_params)
    print(f"Validation score during tuning: {tuned_val_acc:.3f}")

    raw_templates = {k: v[0].astype(float) for k, v in train_set.items()}
    centered_templates = {k: preprocess_image(v[0], (24, 24)).astype(float) for k, v in train_set.items()}
    t2 = time.perf_counter()
    acc_template_raw = evaluate_template_baseline(val_set, raw_templates, centered=False)
    template_raw_ms = (time.perf_counter() - t2) * 1000.0
    t3 = time.perf_counter()
    acc_template_centered = evaluate_template_baseline(val_set, centered_templates, centered=True)
    template_center_ms = (time.perf_counter() - t3) * 1000.0
    t4 = time.perf_counter()
    acc_dt = evaluate_predictor(
        val_set,
        lambda img: distance_transform_baseline_predict(img, centered_templates),
    )
    dt_ms = (time.perf_counter() - t4) * 1000.0
    t5 = time.perf_counter()
    acc_chamfer = evaluate_predictor(
        val_set,
        lambda img: chamfer_baseline_predict(img, centered_templates),
    )
    chamfer_ms = (time.perf_counter() - t5) * 1000.0
    print(f"Baseline template cosine (raw): {acc_template_raw:.3f}")
    print(f"Baseline template cosine (raw) time (ms): {template_raw_ms:.1f}")
    print(f"Baseline template cosine (centered): {acc_template_centered:.3f}")
    print(f"Baseline template cosine (centered) time (ms): {template_center_ms:.1f}")
    print(f"Baseline distance transform: {acc_dt:.3f}")
    print(f"Baseline distance transform time (ms): {dt_ms:.1f}")
    print(f"Baseline chamfer-like: {acc_chamfer:.3f}")
    print(f"Baseline chamfer-like time (ms): {chamfer_ms:.1f}")

    cv_mean, cv_folds = cross_validate_toy(dataset, k=2)
    print(f"CVM toy cross-val mean accuracy (k=2): {cv_mean:.3f}")
    print("CVM toy cross-val fold accuracies:", [round(v, 3) for v in cv_folds])

    digits_res = benchmark_sklearn_digits()
    if digits_res is None:
        print("Digits benchmark: sklearn non disponible")
    else:
        digits_acc, ntest = digits_res
        print(f"Digits benchmark (sklearn, test={ntest}): {digits_acc:.3f}")
        classic = benchmark_sklearn_digits_classic()
        if classic is None:
            print("Digits baselines classiques: indisponibles")
        else:
            print(
                f"Digits k-NN (k=3): {classic['knn_acc']:.3f} "
                f"(time={classic['knn_ms']:.1f} ms)"
            )
            print(
                f"Digits logistic regression: {classic['logreg_acc']:.3f} "
                f"(time={classic['logreg_ms']:.1f} ms)"
            )
            print(
                f"Digits SVM RBF: {classic['svm_acc']:.3f} "
                f"(time={classic['svm_ms']:.1f} ms)"
            )
        gamma_results = gamma_study_digits()
        if gamma_results:
            print("Digits gamma study (disjoint split):")
            for g, a in gamma_results:
                print(f"  gamma={g:.2f} -> acc={a:.3f}")
        gamma3 = gamma_study_digits_three_way()
        if gamma3 is not None:
            print(
                f"Digits gamma study 3-way (val={gamma3['n_val']}, test={gamma3['n_test']}):"
            )
            for name, v, t in gamma3["rows"]:
                print(f"  {name}: val={v:.3f}, test={t:.3f}")
            print(
                f"  best_on_val={gamma3['best_on_val']}, "
                f"best_val={gamma3['best_val_acc']:.3f}, "
                f"test_of_best={gamma3['test_acc_of_best']:.3f}"
            )
        gamma_shift = gamma_study_digits_shifted()
        if gamma_shift is not None:
            print(
                f"Digits shifted gamma study 3-way (val={gamma_shift['n_val']}, test={gamma_shift['n_test']}):"
            )
            for name, v, t in gamma_shift["rows"]:
                print(f"  {name}: val={v:.3f}, test={t:.3f}")
            print(
                f"  best_on_val={gamma_shift['best_on_val']}, "
                f"best_val={gamma_shift['best_val_acc']:.3f}, "
                f"test_of_best={gamma_shift['test_acc_of_best']:.3f}"
            )

    abl = ablation_cvm(train_set, val_set)
    print("CVM ablation:")
    print(
        f"  champ seul={abl['champ_seul']:.3f}, "
        f"+penalite={abl['champ_plus_penalite']:.3f}, "
        f"+transfos={abl['champ_penalite_transfos']:.3f}, "
        f"+topo={abl['champ_penalite_transfos_topo']:.3f}"
    )

    stress_val = build_stress_val_set(dataset)
    abl_stress = ablation_cvm(train_set, stress_val, max_transform_candidates=8)
    print("CVM ablation (stress set):")
    print(
        f"  champ seul={abl_stress['champ_seul']:.3f}, "
        f"+penalite={abl_stress['champ_plus_penalite']:.3f}, "
        f"+transfos={abl_stress['champ_penalite_transfos']:.3f}, "
        f"+topo={abl_stress['champ_penalite_transfos_topo']:.3f}"
    )

    model_fast = train_model(
        train_set,
        shape=(24, 24),
        distance_kind="manhattan",
        use_gamma_local=True,
        gamma_by_class={k: 0.85 for k in train_set.keys()},
    )
    cfg_fast = TransformConfig(
        translations=[(0, 0), (1, 0), (0, 1)],
        rotations_deg=[-12, 0, 12],
        scales=[0.9, 1.0, 1.1],
        shears=[-0.08, 0.0, 0.08],
    )
    t_full = time.perf_counter()
    acc_full, _, _ = evaluate(
        stress_val,
        model_fast,
        cfg_fast,
        lambda_penalty=0.05,
        alpha_pixel=1.0,
        beta_skeleton=0.15,
        beta_topology=0.05,
        score_mode="cosine",
        max_transform_candidates=None,
    )
    full_ms = (time.perf_counter() - t_full) * 1000.0
    t_sel = time.perf_counter()
    acc_sel, _, _ = evaluate(
        stress_val,
        model_fast,
        cfg_fast,
        lambda_penalty=0.05,
        alpha_pixel=1.0,
        beta_skeleton=0.15,
        beta_topology=0.05,
        score_mode="cosine",
        max_transform_candidates=8,
    )
    sel_ms = (time.perf_counter() - t_sel) * 1000.0
    print(
        f"Transform selection stress: full={acc_full:.3f} ({full_ms:.1f} ms) vs "
        f"selected={acc_sel:.3f} ({sel_ms:.1f} ms)"
    )

    levels = {
        "propre": val_set,
        "perturbe": stress_val,
        "ambigu": build_ambiguous_val_set(dataset),
    }
    benchmark_levels(train_set, levels)

    # Interpretability on first misclassification from perturbed level.
    inter_model = train_model(
        train_set,
        shape=(24, 24),
        distance_kind="manhattan",
        use_gamma_local=True,
        gamma_by_class={k: 0.85 for k in train_set.keys()},
    )
    inter_cfg = TransformConfig(
        translations=[(0, 0), (1, 0), (0, 1)],
        rotations_deg=[-8, 0, 8],
        scales=[0.95, 1.0, 1.05],
        shears=[0.0],
    )
    for truth, sample in stress_val:
        pred, _ = predict(
            sample,
            inter_model,
            inter_cfg,
            lambda_penalty=0.05,
            alpha_pixel=1.0,
            beta_skeleton=0.15,
            beta_topology=0.05,
            score_mode="cosine",
            max_transform_candidates=8,
        )
        if pred != truth:
            p, s = explain_prediction(
                sample,
                truth,
                inter_model,
                inter_cfg,
                lambda_penalty=0.05,
                score_mode="cosine",
                max_transform_candidates=8,
                show_plot=show_plot,
            )
            print(f"Interpretability sample: truth={truth}, pred={p}, scores={ {k: round(v,4) for k,v in s.items()} }")
            break

    sample = val_set[0][1]
    p_img = preprocess_image(sample, output_shape=(24, 24))
    skel = zhang_suen_thinning(p_img)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(p_img, cmap="gray_r", interpolation="nearest")
    axes[0].set_title("Preprocessed")
    axes[1].imshow(skel, cmap="gray_r", interpolation="nearest")
    axes[1].set_title("Skeleton")
    axes[2].imshow(tuned_model.fields_pixel["A"], cmap="hot", interpolation="nearest")
    axes[2].set_title("Field A")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    if show_plot:
        plt.show()


if __name__ == "__main__":
    run_demo()
