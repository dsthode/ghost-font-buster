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

Once both layers' velocities are known, the default reveal is
motion-compensated temporal averaging: shift every frame backward by a
candidate layer's cumulative displacement and average the whole stack.
The layer whose motion you compensated for lines back up frame after
frame and reinforces; the other layer, now sliding at roughly double the
relative speed, washes out to a flat tone. Subtracting the plain
(uncompensated) temporal average removes whatever is common to every
frame regardless of alignment (static compression artifacts, vignetting),
leaving just the revealed layer's structure.

That whole-clip approach assumes a layer's own dot pattern keeps its
identity translating for the *entire* clip. On the reference footage that
held for the message layer but not for the plain-noise layer -- its
pairwise frame correlation collapses from strong to negligible within a
few dozen frames, meaning its "noise" isn't one rigid texture but keeps
partially regenerating. A whole-clip average of it just mixes decorrelated
content into noise and buries whatever's embedded there. The fix
(`local_median_reveal`) is to reconstruct from many short, independent
windows short enough to stay inside that layer's coherence length,
MEDIAN-combine each (robust to the other layer contaminating roughly half
of any short window -- a mean isn't), and average the resulting images.
This is in fact how a second, distinct hidden phrase was found on this
footage's plain-noise layer, one the whole-clip method never surfaced.
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
    `velocity` becomes stationary and reinforces; everything else smears.

    This is a one-way scroll, not a looping/tileable texture: each frame,
    fresh content enters from one edge and old content permanently exits
    the other (confirmed by watching the raw footage -- the trailing edge
    does not reappear at the leading edge). So a naive np.roll (wrap-around)
    is wrong: it would splice unrelated rows together at the seam and
    corrupt every frame once the cumulative shift exceeds the frame height.
    Instead, accumulate only over the rows each frame actually has valid
    data for, and normalize by the per-row count of contributing frames
    (which tapers off away from the reference frame, since content nearer
    the exit edge has less time before it scrolls out of view).
    """
    h, w = frames[0].shape
    acc = np.zeros((h, w), dtype=np.float64)
    count = np.zeros((h, w), dtype=np.float64)
    for i, frame in enumerate(frames):
        shift = int(round(i * velocity))
        if shift >= 0:
            if shift >= h:
                continue
            acc[: h - shift, :] += frame[shift:, :]
            count[: h - shift, :] += 1
        else:
            s = -shift
            if s >= h:
                continue
            acc[s:, :] += frame[: h - s, :]
            count[s:, :] += 1
    avg = np.divide(acc, count, out=np.zeros_like(acc), where=count > 0).astype(np.float32)
    return avg, count


MIN_SAMPLE_FRACTION = 0.15  # rows with fewer than this fraction of frames contributing are unreliable


def reveal_layer(
    frames: list[np.ndarray], velocity: float, plain_avg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    aligned, count = motion_compensated_average(frames, velocity)
    diff = aligned - plain_avg
    valid = count >= MIN_SAMPLE_FRACTION * len(frames)
    return diff, valid


DEFAULT_MEDIAN_WINDOW = 4   # half-width in frames of each short reconstruction
DEFAULT_MEDIAN_STRIDE = 3   # frames between successive window centers


def local_median_reveal(
    frames: list[np.ndarray],
    velocity: float,
    window: int = DEFAULT_MEDIAN_WINDOW,
    stride: int = DEFAULT_MEDIAN_STRIDE,
) -> np.ndarray:
    """Alternative reveal method for a layer whose own dot pattern doesn't
    stay coherent for the whole clip -- see the module docstring. Slides a
    small `2*window+1`-frame reconstruction across the clip (motion-
    compensated, no wrap-around, same as `motion_compensated_average`),
    takes the per-pixel MEDIAN of each one, and averages all of those
    together. Median rather than mean within each short window because
    the other, independently-moving layer still contaminates roughly half
    of any short window's frames at a given pixel -- a mean would let
    that contamination through, a median rejects it as long as it's a
    minority of the window. Returns an image, not a diff -- no baseline
    subtraction is needed since the median already suppresses the other
    layer's contribution directly, and it's inverted (bright = ink) to
    match the polarity `enhance` produces from `reveal_layer`.
    """
    h, w = frames[0].shape
    acc = np.zeros((h, w), dtype=np.float64)
    n_windows = 0
    for center in range(window, len(frames) - window, stride):
        idxs = range(center - window, center + window + 1)
        stack = np.full((len(idxs), h, w), np.nan, dtype=np.float32)
        for k, j in enumerate(idxs):
            shift = int(round((j - center) * velocity))
            if shift >= 0:
                if shift >= h:
                    continue
                stack[k, : h - shift, :] = frames[j][shift:, :]
            else:
                s = -shift
                if s >= h:
                    continue
                stack[k, s:, :] = frames[j][: h - s, :]
        local_median = np.nanmedian(stack, axis=0)
        acc += np.nan_to_num(local_median, nan=float(np.nanmean(local_median)))
        n_windows += 1
    if n_windows == 0:
        raise ValueError("clip too short for the requested median window/stride")
    return (255.0 - acc / n_windows).astype(np.float32)


DRIFT_CHUNK_FRAMES = 20     # frames per chunk used to sample the secondary drift
DRIFT_STRIDE = 12           # frames between successive chunk centers
DRIFT_MAX_SHIFT = 250       # search radius (px) for matching chunks against each other
DRIFT_MIN_CONFIDENCE = 0.5  # below this correlation, don't trust/apply the drift estimate
DRIFT_SIGNIFICANCE_PX = 8   # below this much total range, correction isn't worth the cost


def _global_shift(a: np.ndarray, b: np.ndarray, max_shift: int) -> tuple[int, int, float]:
    """Coarse 2D translation of `b` relative to `a`, searched over a wide
    range via cv2.matchTemplate (an optimized C implementation -- a Python
    loop over a range this wide is too slow to be practical). Returns
    (dy, dx, confidence) where positive dy/dx means b's content sits
    below/right of where it is in a. Confidence is the correlation
    coefficient at the match, comparable to the other correlation values
    used throughout this module."""
    a_pad = cv2.copyMakeBorder(a, max_shift, max_shift, max_shift, max_shift, cv2.BORDER_REPLICATE)
    res = cv2.matchTemplate(a_pad, b, cv2.TM_CCOEFF_NORMED)
    _, maxval, _, maxloc = cv2.minMaxLoc(res)
    dy = -(maxloc[1] - max_shift)
    dx = -(maxloc[0] - max_shift)
    return dy, dx, float(maxval)


def estimate_secondary_drift(
    frames: list[np.ndarray],
    velocity: float,
    chunk_frames: int = DRIFT_CHUNK_FRAMES,
    stride: int = DRIFT_STRIDE,
    max_shift: int = DRIFT_MAX_SHIFT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect a slow secondary 2D drift riding on top of a layer's primary
    per-frame velocity -- e.g. a diagonal wander or screensaver-style
    bounce superimposed on the fast scroll. This is far too slow (a
    fraction of a pixel per frame) for the differential frame-to-frame
    block matching that finds the primary velocity to pick up reliably,
    but it integrates to many pixels over the length of a clip, so it
    shows up clearly comparing widely-separated reconstructions instead:
    reconstruct many short, primary-velocity-compensated chunks spanning
    the whole clip, and cross-correlate each against the first with
    `_global_shift`. Returns (centers, dy, dx, confidence) sample arrays,
    one entry per chunk (the first is the reference, all zeros).
    """
    n = len(frames)
    half = chunk_frames // 2
    centers = list(range(half, n - half, stride))
    if len(centers) < 2:
        return np.array(centers), np.zeros(len(centers)), np.zeros(len(centers)), np.ones(len(centers))
    imgs = []
    for c in centers:
        img = local_median_reveal(frames[c - half : c + half], velocity, window=4, stride=2)
        imgs.append(enhance(img, np.ones(img.shape, dtype=bool)))
    dys, dxs, confs = [0.0], [0.0], [1.0]
    for img in imgs[1:]:
        dy, dx, conf = _global_shift(imgs[0], img, max_shift)
        dys.append(float(dy))
        dxs.append(float(dx))
        confs.append(conf)
    return np.array(centers), np.array(dys), np.array(dxs), np.array(confs)


def local_median_reveal_2d(
    frames: list[np.ndarray],
    velocity: float,
    sec_dy: np.ndarray,
    sec_dx: np.ndarray,
    window: int = DEFAULT_MEDIAN_WINDOW,
    stride: int = DEFAULT_MEDIAN_STRIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Like `local_median_reveal`, but also cancels a slow secondary 2D
    drift (per-frame arrays, e.g. from interpolating `estimate_secondary_drift`
    samples). Each short window is still only compensated for the primary
    velocity internally -- the drift's own contribution over such a short
    span is negligible -- but the window's whole reconstructed image is
    then placed on a shared canvas at an offset that cancels *that
    window's* accumulated drift, so windows from every point in the clip
    reinforce the same absolute structure instead of the drift smearing
    them apart. (Naively folding the drift into the intra-window shift
    instead of doing this doesn't work: it cancels out of the within-
    window arithmetic and never actually gets applied.) Returns (image,
    valid) like `reveal_layer`, since the shared canvas is now larger than
    the frame and only partially covered by any given window.
    """
    h, w = frames[0].shape
    pad = int(np.ceil(max(sec_dy.max() - sec_dy.min(), sec_dx.max() - sec_dx.min()))) + 2
    ch, cw = h + 2 * pad, w + 2 * pad
    acc = np.zeros((ch, cw), dtype=np.float64)
    count = np.zeros((ch, cw), dtype=np.float64)
    for center in range(window, len(frames) - window, stride):
        idxs = range(center - window, center + window + 1)
        stack = np.full((len(idxs), h, w), np.nan, dtype=np.float32)
        for k, j in enumerate(idxs):
            shift = int(round((j - center) * velocity))
            if shift >= 0:
                if shift >= h:
                    continue
                stack[k, : h - shift, :] = frames[j][shift:, :]
            else:
                s = -shift
                if s >= h:
                    continue
                stack[k, s:, :] = frames[j][: h - s, :]
        local_median = np.nanmedian(stack, axis=0)
        local_median = np.nan_to_num(local_median, nan=float(np.nanmean(local_median)))
        gy = pad - int(round(sec_dy[center]))
        gx = pad - int(round(sec_dx[center]))
        acc[gy : gy + h, gx : gx + w] += local_median
        count[gy : gy + h, gx : gx + w] += 1
    avg = np.divide(acc, count, out=np.zeros_like(acc), where=count > 0)
    valid = count >= MIN_SAMPLE_FRACTION * count.max() if count.max() > 0 else count > 0
    return (255.0 - avg).astype(np.float32), valid


def enhance(
    diff: np.ndarray, valid: np.ndarray, blur_sigma_x: float = 18.0, blur_sigma_y: float = 6.0
) -> np.ndarray:
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

    Rows near the scroll's exit edge (see `motion_compensated_average`) are
    only averaged over a handful of frames, so they're much noisier than
    the rest of the image -- close to raw single-frame contrast rather than
    the well-averaged interior. Blurring naively would bleed that noise
    into the valid region at the boundary, and including it in the contrast
    stretch would let a few extreme outlier pixels wash out the real
    signal. `valid` marks which pixels are trustworthy; the blur is a
    normalized convolution restricted to them, and the percentile stretch
    ignores everything else.
    """
    maskf = valid.astype(np.float32)
    blurred_signal = cv2.GaussianBlur(diff * maskf, (0, 0), sigmaX=blur_sigma_x, sigmaY=blur_sigma_y)
    blurred_mask = cv2.GaussianBlur(maskf, (0, 0), sigmaX=blur_sigma_x, sigmaY=blur_sigma_y)
    blurred = np.divide(
        blurred_signal, blurred_mask, out=np.zeros_like(blurred_signal), where=blurred_mask > 1e-6
    )
    lo, hi = np.percentile(blurred[valid], [1, 99])
    stretched = np.clip((blurred - lo) / (hi - lo + 1e-6), 0, 1)
    img8 = (stretched * 255).astype(np.uint8)
    img8[~valid] = 128
    return img8


def to_uint8(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [0.5, 99.5])
    out = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
    return (out * 255).astype(np.uint8)


def _resolve_method(method: str, layer_name: str) -> str:
    if method != "auto":
        return method
    # Empirically, on the reference footage the message layer stays
    # coherent for the whole clip (mean works great) while the plain-noise
    # layer doesn't (needs median) -- see module docstring. This is a
    # reasonable default, not a universal law; override with --method if
    # a different clip behaves the other way.
    return "median" if layer_name == "down" else "mean"


def _reveal(
    frames: list[np.ndarray],
    velocity: float,
    plain_avg: np.ndarray,
    method: str,
    drift_correction: str = "auto",
) -> np.ndarray:
    if method == "median":
        if drift_correction != "off":
            centers, sec_dy, sec_dx, confs = estimate_secondary_drift(frames, velocity)
            if len(centers) > 1:
                drift_range = max(sec_dy.max() - sec_dy.min(), sec_dx.max() - sec_dx.min())
                confident = np.median(confs[1:]) >= DRIFT_MIN_CONFIDENCE
                use_drift = confident and (
                    drift_correction == "on" or drift_range >= DRIFT_SIGNIFICANCE_PX
                )
                print(
                    f"secondary drift check: range={drift_range:.1f}px "
                    f"confidence={np.median(confs[1:]):.2f} "
                    f"-> {'applying 2D correction' if use_drift else 'not significant, skipping'}",
                    file=sys.stderr,
                )
                if use_drift:
                    n = len(frames)
                    idx = np.arange(n)
                    interp_dy = np.interp(idx, centers, sec_dy, left=sec_dy[0], right=sec_dy[-1])
                    interp_dx = np.interp(idx, centers, sec_dx, left=sec_dx[0], right=sec_dx[-1])
                    img, valid = local_median_reveal_2d(frames, velocity, interp_dy, interp_dx)
                    return enhance(img, valid)
        img = local_median_reveal(frames, velocity)
        return enhance(img, np.ones(img.shape, dtype=bool))
    diff, valid = reveal_layer(frames, velocity, plain_avg)
    return enhance(diff, valid)


def run(
    video_path: str,
    out_path: str,
    layer: str = "both",
    max_shift: int = DEFAULT_MAX_SHIFT,
    method: str = "auto",
    drift_correction: str = "auto",
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

    def build(v: float, layer_name: str) -> np.ndarray:
        m = _resolve_method(method, layer_name)
        print(f"revealing {layer_name} layer with method={m}", file=sys.stderr)
        return _reveal(frames, v, plain_avg, m, drift_correction=drift_correction)

    if layer == "both":
        base, ext = out_path.rsplit(".", 1) if "." in out_path else (out_path, "png")
        up_img = build(up_est.velocity, "up")
        down_img = build(down_est.velocity, "down")
        cv2.imwrite(f"{base}_up.{ext}", up_img)
        cv2.imwrite(f"{base}_down.{ext}", down_img)
        print(f"wrote {base}_up.{ext} and {base}_down.{ext}", file=sys.stderr)
    else:
        v = up_est.velocity if layer == "up" else down_est.velocity
        cv2.imwrite(out_path, build(v, layer))
        print(f"wrote {out_path}", file=sys.stderr)

    if save_diagnostics:
        base, ext = out_path.rsplit(".", 1) if "." in out_path else (out_path, "png")
        cv2.imwrite(f"{base}_diag_plain_average.{ext}", to_uint8(plain_avg))
        up_diff, up_valid = reveal_layer(frames, up_est.velocity, plain_avg)
        down_diff, down_valid = reveal_layer(frames, down_est.velocity, plain_avg)
        up_diff[~up_valid] = 0
        down_diff[~down_valid] = 0
        cv2.imwrite(f"{base}_diag_raw_diff_up.{ext}", to_uint8(up_diff))
        cv2.imwrite(f"{base}_diag_raw_diff_down.{ext}", to_uint8(down_diff))
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
        default="both",
        help="which motion layer to reveal (default: both -- on the reference "
        "footage each layer carries a different, independent hidden phrase)",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "mean", "median"],
        default="auto",
        help="reveal algorithm: 'mean' is whole-clip motion-compensated averaging "
        "(reveal_layer), 'median' is the short-window median approach "
        "(local_median_reveal) for a layer whose own dot pattern doesn't stay "
        "coherent for the whole clip, 'auto' (default) picks mean for the "
        "upward layer and median for the downward one, matching what worked "
        "on the reference footage",
    )
    parser.add_argument(
        "--max-shift",
        type=int,
        default=DEFAULT_MAX_SHIFT,
        help=f"maximum per-frame pixel shift to search for (default: {DEFAULT_MAX_SHIFT})",
    )
    parser.add_argument(
        "--drift-correction",
        choices=["auto", "on", "off"],
        default="auto",
        help="correct for a slow secondary 2D drift on top of the primary velocity "
        "(a diagonal wander or screensaver-style bounce, too slow for the primary "
        "velocity estimate to see but which integrates to a real, visible smear over "
        "the clip) -- only applies to the median method. 'auto' (default) detects "
        "and corrects it only if the measured drift is confidently large enough to "
        "matter; 'on' always applies it if a confident measurement exists; 'off' "
        "disables the check entirely",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="also save the plain temporal average and un-enhanced mean-method diffs "
        "for both layers",
    )
    args = parser.parse_args()
    run(args.video, args.output, layer=args.layer, max_shift=args.max_shift,
        method=args.method, drift_correction=args.drift_correction,
        save_diagnostics=args.diagnostics)


if __name__ == "__main__":
    main()
