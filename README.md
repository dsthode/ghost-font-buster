# ghost-font-buster

Recovers text hidden inside a "ghost font" style video: two overlaid
random-dot noise layers, one plain and one with its dot density shaped
into letters, each translating rigidly in opposite directions (e.g. noise
scrolling down, message scrolling up). In any single frame it just looks
like static — the message only exists as a *motion* signal, invisible to
a human glance and to frame-based OCR/vision models alike.

`samples/ghostmessage.mp4` is the reference clip this was built and tested
against. Running the buster on it reveals: **"WRITTEN IN GHOST FONT"**.

## Why not plain optical flow?

The first instinct is `cv2.calcOpticalFlowFarneback`. It doesn't work here
— run with `--diagnostics` and you can confirm the dense flow field comes
back essentially zero. Gradient-based flow (Farneback, Lucas-Kanade) relies
on local image gradients being consistent enough to disambiguate a match
between frames. On i.i.d. random-dot noise, every neighborhood looks like
every other neighborhood, so there's no reliable local gradient to follow
— it's the aperture problem in its worst form. This is also why the motion
is imperceptible to a human watching casually: your visual system needs to
integrate over many frames globally to pick up the coherent drift, the
same thing this script automates.

What does work is **global cross-correlation / block matching**: shift a
whole frame by a candidate vertical offset and correlate it against the
next frame across the entire image. True motion in a rigid noise field only
correlates well at the true shift; because two independently-moving layers
are superimposed, the correlation-vs-shift curve has two peaks — one
positive (the layer moving down), one negative (the layer moving up).

## How the reveal works

1. **Estimate both layers' velocities.** For every consecutive frame pair,
   sweep candidate vertical shifts and cross-correlate the whole frame
   (`estimate_layer_velocities`). Take the best positive-shift peak and
   best negative-shift peak per pair (subpixel-refined via a parabola fit),
   then combine across the clip with a median — robust to the odd
   stalled/duplicated frame some encoders emit at the start.
2. **Motion-compensate and average.** For a candidate layer velocity `v`,
   shift frame `i` backward by `round(i * v)` pixels and average the whole
   stack (`motion_compensated_average`). Content moving at `v` lines back
   up every frame and reinforces; the other layer, now sliding at roughly
   double the relative speed, decorrelates frame to frame and washes out
   to a flat gray. This is a one-way scroll, not a looping/tileable
   texture — fresh content enters from one edge each frame and old content
   permanently exits the other, confirmed by inspecting the raw footage.
   So the shift is applied with each frame contributing only to the rows
   it actually has data for (no wrap-around), and each row is normalized
   by how many frames actually covered it. That count necessarily tapers
   off away from the reference frame — content near a layer's exit edge
   has less time on screen before it scrolls out of view — so rows near
   the edge of the output are averaged over far fewer samples than the
   interior and are noticeably noisier as a result.
3. **Subtract the plain temporal average.** Averaging the raw,
   unaligned frames captures whatever is identical in every frame
   regardless of alignment — static compression grain, vignetting.
   Subtracting it out leaves just the structure the alignment step
   revealed.
4. **Denoise anisotropically and stretch contrast** (`enhance`). What's
   left after step 3 is mostly clean, but has a thin, high-frequency
   vertical-grain residue. A blur that's wider horizontally than
   vertically kills that grain (it's much finer than a letter) while
   barely softening the letters themselves, followed by a plain percentile
   contrast stretch. Both the blur and the percentile range are computed
   as a normalized convolution over just the well-sampled rows from step
   2 (see `MIN_SAMPLE_FRACTION`) — otherwise the noisy, low-sample edge
   rows would bleed into the blur and skew the contrast stretch. Those
   edge rows are rendered as flat mid-gray in the output instead.

## Usage

```bash
pip install -r requirements.txt

# Reveal the upward-moving layer (the default -- matches "noise moves
# down, message moves up")
python ghost_font_buster.py samples/ghostmessage.mp4 -o revealed.png

# Not sure which direction carries the message? Get both.
python ghost_font_buster.py samples/ghostmessage.mp4 -o revealed.png --layer both

# Also dump the plain average and raw (pre-enhancement) diffs for both
# layers, to see the pipeline's intermediate state.
python ghost_font_buster.py samples/ghostmessage.mp4 -o revealed.png --diagnostics
```

Options:

- `--layer {up,down,both}` — which motion direction to reconstruct
  (default `up`).
- `--max-shift N` — widen the per-frame search range if your clip's motion
  is faster than the default ±40 px/frame.
- `--diagnostics` — also write the plain temporal average and both
  layers' un-enhanced diffs, useful for tuning or for confirming the
  velocity estimate on a new clip.

The script prints the detected per-frame velocity (and its
frame-to-frame standard deviation) for both layers to stderr — a high
standard deviation is a sign the source isn't a constant-velocity
translation and reconstruction quality may suffer, since the core
algorithm assumes constant velocity per layer.
