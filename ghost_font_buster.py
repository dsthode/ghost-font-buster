#!/usr/bin/env python3
"""Recover text hidden in a two-layer random-dot kinematogram video.

The trick this script defeats: two random-dot noise fields are overlaid on
each other. One field (the "carrier") is plain noise and translates
rigidly in one direction (e.g. downward). The other field (the "message")
is noise too, but its dot density is modulated by a text mask, and it
translates rigidly in the opposite direction (e.g. upward). Any single
frame looks like uniform noise to both a human and an OCR model -- the
message has no static contrast, only a *motion* signature.

Why not just run cv2.calcOpticalFlowFarneback and call it done? Because
this content is specifically built to defeat local, gradient-based flow.
Every pixel neighbourhood in i.i.d. random-dot noise looks the same, so
there is no local gradient consistent enough to disambiguate a match: it's
the aperture problem taken to its worst case. (This is the same reason
humans can't consciously perceive the motion frame-to-frame either -- it
takes many frames of global integration.) Dense optical flow measured on
this footage with cv2.calcOpticalFlowFarneback comes back essentially zero
everywhere -- try it yourself, it's a quick sanity check.

What does work is global cross-correlation / block matching: for a
candidate vertical shift, translate one frame and correlate it against the
next across the *whole* image. Random noise only correlates well at the
true shift, and because two rigid layers are superimposed, the correlation
curve has two peaks -- one per layer, one positive (moving down) and one
negative (moving up). This is still a motion/flow-estimation technique
(equivalent to phase correlation / PIV block matching), just one that
integrates over a large window instead of a differential neighbourhood.

Once both layers' velocities are known, the reveal is motion-compensated
temporal averaging: shift every frame backward by a candidate layer's
cumulative displacement and average the whole stack. The layer whose
motion you compensated for lines back up frame after frame and reinforces;
the other layer, now sliding at roughly double the relative speed, washes
out to a flat tone (dots at any fixed compensated pixel become an
uncorrelated sample from frame to frame). Subtracting the plain
(uncompensated) temporal average removes whatever is common to every
frame regardless of alignment (static compression artifacts, vignetting),
leaving just the revealed layer's structure.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_MAX_SHIFT = 40
DEFAULT_CROP = 30
DUPLICATE_FRAME_THRESHOLD = 1.0  # mean abs diff below this => treat pair as a stall, skip


@dataclass
class LayerEstimate:
    velocity: float          # median px/frame, signed (negative = moving up the frame)
    step_std: float          # std of the per-frame-pair estimates velocity was medianed from
    n_samples: int


def load_gray_frames(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"could not open video: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    if len(frames) < 3:
        raise ValueError(f"video only has {len(frames)} frames, need at least 3")
    return frames


def _vertical_xcorr_curve(
    f0: np.ndarray, f1: np.ndarray, max_shift: int, crop: int
) -> tuple[np.ndarray, np.ndarray]:
    """Normalized cross-correlation of f0 against f1 for vertical shifts in
    [-max_shift, max_shift]. This is a brute-force block match: for a
    random-noise field it is far more reliable than differential flow."""
    f0c = f0 - f0.mean()
    f1c = f1 - f1.mean()
    h, _w = f0.shape
    shifts = np.arange(-max_shift, max_shift + 1)
    corrs = np.empty(len(shifts), dtype=np.float64)
    for idx, dy in enumerate(shifts):
        if dy >= 0:
            a = f0c[crop : h - crop - dy, :]
            b = f1c[crop + dy : h - crop, :]
        else:
            a = f0c[crop - dy : h - crop, :]
            b = f1c[crop : h - crop + dy, :]
        corrs[idx] = np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return shifts, corrs


def _subpixel_peak(shifts: np.ndarray, corrs: np.ndarray, lo: int, hi: int) -> tuple[float, float]:
    """Argmax within [lo, hi], refined to subpixel precision via a parabola
    fit through the peak and its two neighbours."""
    mask = (shifts >= lo) & (shifts <= hi)
    idxs = np.where(mask)[0]
    imax = idxs[np.argmax(corrs[idxs])]
    if imax <= 0 or imax >= len(corrs) - 1:
        return float(shifts[imax]), float(corrs[imax])
    y0, y1, y2 = corrs[imax - 1], corrs[imax], corrs[imax + 1]
    denom = y0 - 2 * y1 + y2
    delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
    delta = max(-1.0, min(1.0, delta))
    return float(shifts[imax]) + delta, float(y1)


def estimate_layer_velocities(
    frames: list[np.ndarray], max_shift: int = DEFAULT_MAX_SHIFT, crop: int = DEFAULT_CROP
) -> tuple[LayerEstimate, LayerEstimate]:
    """Estimate the (signed, px/frame) velocity of each of the two motion
    layers by block-matching every consecutive frame pair and taking the
    dominant positive-shift and negative-shift correlation peak of each
    pair, then combining across the whole clip with a median (robust to
    the odd stalled/duplicated frame some encoders emit at the start)."""
    up_steps = []
    down_steps = []
    for i in range(len(frames) - 1):
        f0, f1 = frames[i], frames[i + 1]
        if np.abs(f0 - f1).mean() < DUPLICATE_FRAME_THRESHOLD:
            continue  # stalled/duplicated frame pair carries no motion signal
        shifts, corrs = _vertical_xcorr_curve(f0, f1, max_shift, crop)
        down_peak, _ = _subpixel_peak(shifts, corrs, 1, max_shift)
        up_peak, _ = _subpixel_peak(shifts, corrs, -max_shift, -1)
        down_steps.append(down_peak)
        up_steps.append(up_peak)

    if not up_steps:
        raise RuntimeError("no usable frame pairs found to estimate motion from")

    up_steps = np.array(up_steps)
    down_steps = np.array(down_steps)
    up = LayerEstimate(float(np.median(up_steps)), float(np.std(up_steps)), len(up_steps))
    down = LayerEstimate(float(np.median(down_steps)), float(np.std(down_steps)), len(down_steps))
    return up, down


def motion_compensated_average(frames: list[np.ndarray], velocity: float) -> np.ndarray:
    """Average every frame after undoing `velocity` px/frame of vertical
    translation, using frame 0 as the reference. Layer content moving at
    `velocity` becomes stationary and reinforces; everything else smears."""
    h, w = frames[0].shape
    acc = np.zeros((h, w), dtype=np.float64)
    for i, frame in enumerate(frames):
        shift = int(round(i * velocity))
        # np.roll (wrap-around) rather than a zero/replicate border: the
        # dot fields tile seamlessly, confirmed empirically by alignment
        # quality at large frame offsets being just as good as at small ones.
        acc += np.roll(frame, -shift, axis=0)
    return (acc / len(frames)).astype(np.float32)


def reveal_layer(frames: list[np.ndarray], velocity: float, plain_avg: np.ndarray) -> np.ndarray:
    aligned = motion_compensated_average(frames, velocity)
    return aligned - plain_avg


def enhance(diff: np.ndarray, blur_sigma_x: float = 18.0, blur_sigma_y: float = 6.0) -> np.ndarray:
    """Denoise and contrast-stretch a revealed-layer image for readability.

    The residual noise left after motion-compensated averaging is a thin,
    high-frequency vertical texture (leftover dot/compression grain), while
    letters are large, low-frequency blobs. An anisotropic blur (wider
    horizontally) averages the grain toward flat gray while barely
    softening the letters, which is a much better trade than an isotropic
    blur of the same overall strength. A plain percentile contrast stretch
    is enough after that; CLAHE was tried and rejected here -- its local
    tiling re-amplifies exactly the fine-grained residual grain the blur
    just suppressed, producing blotchy artifacts instead of a cleaner image.
    """
    blurred = cv2.GaussianBlur(diff, (0, 0), sigmaX=blur_sigma_x, sigmaY=blur_sigma_y)
    lo, hi = np.percentile(blurred, [1, 99])
    stretched = np.clip((blurred - lo) / (hi - lo + 1e-6), 0, 1)
    return (stretched * 255).astype(np.uint8)


def to_uint8(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [0.5, 99.5])
    out = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
    return (out * 255).astype(np.uint8)


def run(
    video_path: str,
    out_path: str,
    layer: str = "up",
    max_shift: int = DEFAULT_MAX_SHIFT,
    save_diagnostics: bool = False,
) -> None:
    frames = load_gray_frames(video_path)
    print(f"loaded {len(frames)} frames from {video_path}", file=sys.stderr)

    up_est, down_est = estimate_layer_velocities(frames, max_shift=max_shift)
    print(
        f"detected layer velocities (px/frame): "
        f"up={up_est.velocity:+.3f} (std {up_est.step_std:.3f}, n={up_est.n_samples}), "
        f"down={down_est.velocity:+.3f} (std {down_est.step_std:.3f}, n={down_est.n_samples})",
        file=sys.stderr,
    )
    for est, name in ((up_est, "up"), (down_est, "down")):
        if est.step_std > 0.75:
            print(
                f"warning: {name}-layer per-frame shift is noisy (std={est.step_std:.2f}); "
                "velocity may not be constant, reconstruction quality may suffer",
                file=sys.stderr,
            )

    plain_avg = np.mean(np.stack(frames), axis=0)

    def build(v: float) -> np.ndarray:
        return enhance(reveal_layer(frames, v, plain_avg))

    if layer == "both":
        base, ext = out_path.rsplit(".", 1) if "." in out_path else (out_path, "png")
        up_img = build(up_est.velocity)
        down_img = build(down_est.velocity)
        cv2.imwrite(f"{base}_up.{ext}", up_img)
        cv2.imwrite(f"{base}_down.{ext}", down_img)
        print(f"wrote {base}_up.{ext} and {base}_down.{ext}", file=sys.stderr)
    else:
        v = up_est.velocity if layer == "up" else down_est.velocity
        cv2.imwrite(out_path, build(v))
        print(f"wrote {out_path}", file=sys.stderr)

    if save_diagnostics:
        base, ext = out_path.rsplit(".", 1) if "." in out_path else (out_path, "png")
        cv2.imwrite(f"{base}_diag_plain_average.{ext}", to_uint8(plain_avg))
        cv2.imwrite(
            f"{base}_diag_raw_diff_up.{ext}",
            to_uint8(reveal_layer(frames, up_est.velocity, plain_avg)),
        )
        cv2.imwrite(
            f"{base}_diag_raw_diff_down.{ext}",
            to_uint8(reveal_layer(frames, down_est.velocity, plain_avg)),
        )
        print(f"wrote diagnostic images alongside {out_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover text hidden in a two-layer random-dot kinematogram video."
    )
    parser.add_argument("video", help="path to the input video")
    parser.add_argument(
        "-o", "--output", default="revealed.png", help="output image path (default: revealed.png)"
    )
    parser.add_argument(
        "--layer",
        choices=["up", "down", "both"],
        default="up",
        help="which motion layer to reveal: the one translating upward, downward, "
        "or both (default: up, matching the described 'message moves up, "
        "noise moves down' construction)",
    )
    parser.add_argument(
        "--max-shift",
        type=int,
        default=DEFAULT_MAX_SHIFT,
        help=f"maximum per-frame pixel shift to search for (default: {DEFAULT_MAX_SHIFT})",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="also save the plain temporal average and un-enhanced diffs for both layers",
    )
    args = parser.parse_args()
    run(args.video, args.output, layer=args.layer, max_shift=args.max_shift,
        save_diagnostics=args.diagnostics)


if __name__ == "__main__":
    main()
