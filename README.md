# ghost-font-buster

Recovers text hidden inside a "ghost font" style video: two overlaid
random-dot noise layers, each with its dot density shaped into letters,
translating rigidly in opposite directions (noise scrolling down, message
scrolling up). In any single frame it just looks like static — the
message only exists as a *motion* signal, invisible to a human glance and
to frame-based OCR/vision models alike.

`samples/ghostmessage.mp4` is the reference clip this was built and tested
against. Running the buster on it reveals **two independent messages, one
per layer**:

- upward layer: **"WRITTEN IN GHOST FONT"** — a decoy aimed at whatever's
  doing the reading (see below).
- downward layer: **"HELLO HUMAN"** — the real payload.

`samples/ghostmessage_2.mp4` and `samples/ghostmessage_3.mp4` are two more
clips used to validate the tool against fresh, unseen input rather than
whatever it was originally tuned on. Both hold up: same decoy on the
upward layer, different real payloads on the downward layer ("CLAUDE IS
AWESOME" and "GHOST FONT IS BUSTED" respectively) decoded correctly with
the same default settings, no per-clip tuning. The third clip is also
what surfaced the secondary-drift correction below.

Don't stop at the first phrase a reveal produces. This tool found "WRITTEN
IN GHOST FONT" first too, and it read as a plausible, complete, in-context
answer — it's a label a "ghost font" demo would plausibly show, so there
was no obvious reason to doubt it. It took being told outright that it
wasn't the real message before digging further turned up "HELLO HUMAN" on
the other layer, sitting the whole time behind a bug in the first
attempt at reading that layer. Both layers' output should be treated as
a first draft, not a confirmed answer, until you've looked for a second
phrase and found none.

## Why not plain optical flow?

The first instinct is `cv2.calcOpticalFlowFarneback`. It doesn't work here
— gradient-based flow (Farneback, Lucas-Kanade) relies on local image
gradients being consistent enough to disambiguate a match between frames.
On i.i.d. random-dot noise, every neighborhood looks like every other
neighborhood, so there's no reliable local gradient to follow — it's the
aperture problem in its worst form. Try it yourself on this footage as a
quick sanity check: the dense flow field comes back essentially zero
everywhere. This is also why the motion is imperceptible to a human
watching casually: your visual system needs to integrate over many frames
globally to pick up the coherent drift, the same thing this script
automates.

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
2. **Reveal each layer** with one of two methods (see below): whole-clip
   motion-compensated averaging (`reveal_layer` /
   `motion_compensated_average`), or short-window median reconstruction
   (`local_median_reveal`).
3. **Denoise anisotropically and stretch contrast** (`enhance`). What's
   left after step 2 has a thin, high-frequency vertical-grain residue —
   leftover dot/compression grain — while letters are large, low-frequency
   blobs. A blur that's wider horizontally than vertically kills that grain
   while barely softening the letters, followed by a percentile contrast
   stretch.

### Method 1: whole-clip motion-compensated averaging

Shift frame `i` backward by `round(i * v)` pixels and average the whole
stack. Content moving at `v` lines back up every frame and reinforces;
the other layer, now sliding at roughly double the relative speed,
decorrelates frame to frame and washes out to a flat gray. Subtracting the
plain (unaligned) temporal average removes whatever is common to every
frame regardless of alignment (static compression grain, vignetting),
leaving just the revealed layer's structure.

This is a one-way scroll, not a looping/tileable texture — fresh content
enters from one edge each frame and old content permanently exits the
other, confirmed by inspecting the raw footage. So the shift is applied
with each frame contributing only to the rows it actually has data for (no
wrap-around), and each row is normalized by how many frames actually
covered it. That count necessarily tapers off away from the reference
frame — content near a layer's exit edge has less time on screen before it
scrolls out of view — so rows near the edge of the output are averaged
over far fewer samples than the interior and are noticeably noisier as a
result (`MIN_SAMPLE_FRACTION` marks these unreliable and excludes them
from the contrast stretch, rendering them flat mid-gray instead).

This method assumes a layer's own dot pattern keeps its identity
translating for the *entire* clip. That held for the upward/message layer
on the reference footage but not for the downward/plain-noise layer: its
pairwise frame correlation collapses from strong (~0.6) at 10 frames apart
to negligible (~0.1) by 60 frames apart — it isn't one rigid texture, it
keeps partially regenerating. A whole-clip average of it just mixes
decorrelated, unrelated content into noise and buries whatever's embedded
there ("HELLO HUMAN" is invisible with this method on the downward layer).

### Method 2: short-window median reconstruction

For a layer that doesn't stay coherent for the whole clip, reconstruct
from many short, independent windows instead — short enough to stay
inside that layer's actual coherence length. Each window
(`2*window+1` frames, `DEFAULT_MEDIAN_WINDOW=4`) is motion-compensated
the same no-wraparound way, then combined with a per-pixel MEDIAN rather
than a mean: within a short window the other, independently-moving layer
still contaminates roughly half of any given pixel's frames, and a median
is robust to that where a mean isn't. Slide this across the clip
(`DEFAULT_MEDIAN_STRIDE=3`) and average the resulting images together —
each window is a low-noise independent estimate of the same underlying
structure, so this cancels their remaining independent noise while
reinforcing the shared signal.

This is how "HELLO HUMAN" was found: many independent short windows,
centered at very different points across the whole clip, all
independently agreed on the same phrase — strong evidence it's real
structure and not an artifact of any one window's alignment.

### Secondary drift correction

The downward layer's per-frame velocity estimate isn't the whole motion
story. On every reference clip tested, riding on top of that fast
constant scroll is a much slower secondary 2D drift — a diagonal wander
that on one clip visibly reads as a screensaver-style bounce (a viewer
caught it by eye; it's the reason "GHOST FONT IS BUSTED" first came out
too crowded/overlapping to read cleanly). It's roughly 100px of total
range over the length of a clip but only a fraction of a pixel per frame,
so it's invisible to the differential frame-to-frame block matching that
finds the primary velocity — that noise floor is close to a full pixel of
per-step jitter, which swallows it completely.

It shows up instead by comparing reconstructions from *widely-separated*
points in the clip, where the drift has had time to integrate into
something large enough to see (`estimate_secondary_drift`): rebuild many
short, primary-velocity-compensated chunks spanning the clip, and
cross-correlate each against the first over a wide 2D search range
(`cv2.matchTemplate`, needed for speed — a brute-force Python sweep over a
range this wide is too slow to be practical). The result is a clean,
smooth, high-confidence (>0.9 correlation) trajectory sample every few
chunks.

Correcting for it (`local_median_reveal_2d`) means each short reveal
window is still only compensated for the *primary* velocity internally —
the drift's own contribution over such a short span is negligible — but
the window's whole output image is then placed on a shared, padded canvas
at a global offset that cancels *that window's own* accumulated drift.
(The first version of this tried folding the drift correction into the
existing intra-window shift arithmetic instead; it's a dead end -- the
correction cancels itself out of that arithmetic and never actually gets
applied, since intra-window alignment is inherently relative to the
window's own center regardless of where that center sits on the drift
curve.) That way windows from every point in the clip reinforce the same
absolute structure instead of the drift slowly smearing them apart.

This runs automatically (`--drift-correction on`, the default) whenever
the `median` method is used. Applying it tightened up every message on
every reference clip tested, including the two that nobody had reported
visible diagonal motion on before checking — it isn't a one-off property
of a single clip, it appears to be inherent to how this whole format of
video is generated, so it defaults to *always* applying whenever a
confident correlation measurement exists (`DRIFT_MIN_CONFIDENCE`) rather
than gating on the measured drift also being large (that gate is still
available as `--drift-correction auto`, for a clip where you specifically
want to avoid the extra reconstruction cost unless it's worth it). Use
`--drift-correction off` to skip the check entirely.

## Usage

```bash
pip install -r requirements.txt

# Reveal both layers (the default) -- each with the method that suits it
python ghost_font_buster.py samples/ghostmessage.mp4 -o revealed.png

# Just one direction
python ghost_font_buster.py samples/ghostmessage.mp4 -o revealed.png --layer up

# Force a specific method rather than the up=mean/down=median default
python ghost_font_buster.py samples/ghostmessage.mp4 -o revealed.png --layer down --method mean

# Also dump the plain average and raw (pre-enhancement) mean-method diffs
# for both layers, to see the pipeline's intermediate state.
python ghost_font_buster.py samples/ghostmessage.mp4 -o revealed.png --diagnostics
```

Options:

- `--layer {up,down,both}` — which motion direction to reconstruct
  (default `both`).
- `--method {auto,mean,median}` — `mean` is whole-clip motion-compensated
  averaging, `median` is the short-window approach, `auto` (default) picks
  mean for the upward layer and median for the downward one, matching what
  worked on the reference footage. This is an empirical default for this
  clip, not a universal rule — if a different clip's layers behave the
  other way around, override it.
- `--max-shift N` — widen the per-frame search range if your clip's motion
  is faster than the default ±40 px/frame.
- `--drift-correction {auto,on,off}` — secondary 2D drift correction for
  the median method, see above (default `on`).
- `--diagnostics` — also write the plain temporal average and both
  layers' un-enhanced mean-method diffs, useful for tuning or for
  confirming the velocity estimate on a new clip.

The script prints the detected per-frame velocity (and its
frame-to-frame standard deviation) for both layers to stderr — a high
standard deviation is a sign the source isn't a constant-velocity
translation and reconstruction quality may suffer, since the core
algorithm assumes constant velocity per layer.
