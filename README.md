# ghost-font-buster

Recovers text hidden inside a "ghost font" style video using motion
analysis — no OCR, no ML model, just cross-correlation and statistics.

## Origins

The technique this defeats comes from [mixfont.com/ghost-font](https://www.mixfont.com/ghost-font),
which demonstrates a way to encode text into a video so that it's
(claimed to be) legible to a human watching but indiscernible to an AI
looking at individual frames. The pitch is essentially a captcha inverted:
instead of distorting text so *only* a human can read it, the message
never exists as static contrast anywhere — it's built entirely out of
*motion*, a channel single-frame vision models don't see and single-frame
OCR has nothing to grab onto.

This repo's origin is a straightforward hypothesis about that claim: if
the message is encoded as motion, then a machine vision technique that
operates on motion — rather than on any single frame — should be able to
recover it just as well as a human eye can, defeating the "unreadable to
AI" half of the pitch. That hypothesis is what got tested here, and it
held up: every sample clip decodes cleanly with the methods below, no
manual tuning per clip.

## The mechanism

Each video overlays two random-dot fields, the same trick used in
motion-perception vision science as a "random-dot kinematogram" (RDK):

- A **plain noise layer** with uniform dot density, translating rigidly in
  one direction (e.g. scrolling down).
- A **message layer**, statistically identical noise except its dot
  density is shaped into letterforms, translating rigidly in the
  *opposite* direction (e.g. scrolling up).

Composited together (darkest-pixel-wins), any single frame is just
uniform static. There's no edge, no contrast blob, nothing a frame-based
classifier or OCR pass could latch onto — confirmed by running
`cv2.calcOpticalFlowFarneback` on this footage, which comes back
essentially zero everywhere too (see below for why). The letterforms only
become statistically visible once you integrate structure across many
frames along the *correct* motion trajectory, which is exactly the
capability this tool implements.

`samples/ghostmessage.mp4` is the first clip pulled from the site and
decodes to **two independent messages, one per layer**:

- upward layer: **"WRITTEN IN GHOST FONT"** — a decoy aimed at whatever's
  doing the reading (see "don't stop at the first phrase" below).
- downward layer: **"HELLO HUMAN"** — the real payload.

`samples/ghostmessage_2.mp4` and `samples/ghostmessage_3.mp4` are two more
clips used to validate the tool against fresh, unseen input rather than
whatever it was originally tuned on. Both hold up: same decoy on the
upward layer, different real payloads on the downward layer ("CLAUDE IS
AWESOME" and "GHOST FONT IS BUSTED" respectively) decoded correctly with
the same default settings, no per-clip tuning. The third clip is also
what surfaced the secondary-drift correction described below.

Don't stop at the first phrase a reveal produces. This tool found "WRITTEN
IN GHOST FONT" first too, and it read as a plausible, complete, in-context
answer — it's a label a "ghost font" demo would plausibly show, so there
was no obvious reason to doubt it. It took being told outright that it
wasn't the real message before digging further turned up "HELLO HUMAN" on
the other layer, sitting the whole time behind a bug in the first attempt
at reading that layer. Treat both layers' output as a first draft, not a
confirmed answer, until you've looked for a second phrase and found none.

## Algorithms used

### 1. Why not plain optical flow

The first instinct is `cv2.calcOpticalFlowFarneback`. It doesn't work here
— gradient-based flow (Farneback, Lucas-Kanade) relies on local image
gradients being consistent enough to disambiguate a match between frames.
On i.i.d. random-dot noise, every neighborhood looks like every other
neighborhood, so there's no reliable local gradient to follow — it's the
aperture problem in its worst form. This is also, not coincidentally, why
the motion is hard to consciously register just watching the clip: your
visual system needs to integrate over many frames globally to pick up the
coherent drift, the same thing this tool automates numerically.

### 2. Velocity estimation via global cross-correlation

What does work is **global cross-correlation / block matching**
(`estimate_layer_velocities`): shift a whole frame by a candidate vertical
offset and correlate it against the next frame across the entire image.
True motion in a rigid noise field only correlates well at the true
shift, and because two independently-moving layers are superimposed, the
correlation-vs-shift curve has two peaks — one positive (the layer moving
down), one negative (the layer moving up). Each pair's best peak is
subpixel-refined with a parabola fit, then combined across the whole clip
with a median, which is robust to the odd stalled/duplicated frame some
encoders emit at the start.

### 3. Reveal: whole-clip motion-compensated averaging

`reveal_layer` / `motion_compensated_average`: shift frame `i` backward
by `round(i * v)` pixels and average the whole stack. Content moving at
`v` lines back up every frame and reinforces; the other layer, now
sliding at roughly double the relative speed, decorrelates frame to
frame and washes out to a flat gray. Subtracting the plain (unaligned)
temporal average removes whatever is common to every frame regardless of
alignment (static compression grain, vignetting), leaving just the
revealed layer's structure.

This is a one-way scroll, not a looping/tileable texture — fresh content
enters from one edge each frame and old content permanently exits the
other, confirmed by inspecting the raw footage. So the shift is applied
with each frame contributing only to the rows it actually has data for
(no wrap-around), and each row is normalized by how many frames actually
covered it. That count necessarily tapers off away from the reference
frame — content near a layer's exit edge has less time on screen before
it scrolls out of view — so rows near the edge of the output are averaged
over far fewer samples than the interior and are noticeably noisier
(`MIN_SAMPLE_FRACTION` marks these unreliable and excludes them from the
contrast stretch, rendering them flat mid-gray instead).

This method assumes a layer's own dot pattern keeps its identity
translating for the *entire* clip. That held for the upward/message layer
on the reference footage but not for the downward/plain-noise layer: its
pairwise frame correlation collapses from strong (~0.6) at 10 frames
apart to negligible (~0.1) by 60 frames apart — it isn't one rigid
texture, it keeps partially regenerating. A whole-clip average of it just
mixes decorrelated, unrelated content into noise and buries whatever's
embedded there ("HELLO HUMAN" is invisible with this method on the
downward layer).

### 4. Reveal: short-window median reconstruction

`local_median_reveal`: for a layer that doesn't stay coherent for the
whole clip, reconstruct from many short, independent windows instead —
short enough to stay inside that layer's actual coherence length. Each
window (`2*window+1` frames, `DEFAULT_MEDIAN_WINDOW=4`) is
motion-compensated the same no-wraparound way, then combined with a
per-pixel MEDIAN rather than a mean: within a short window the other,
independently-moving layer still contaminates roughly half of any given
pixel's frames, and a median is robust to that where a mean isn't. Slide
this across the clip (`DEFAULT_MEDIAN_STRIDE=3`) and average the
resulting images together — each window is a low-noise independent
estimate of the same underlying structure, so this cancels their
remaining independent noise while reinforcing the shared signal.

This is how "HELLO HUMAN" was found: many independent short windows,
centered at very different points across the whole clip, all
independently agreed on the same phrase — strong evidence it's real
structure and not an artifact of any one window's alignment.

### 5. Secondary 2D drift correction

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
existing intra-window shift arithmetic instead; it's a dead end — the
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

### 6. Denoise and contrast stretch

`enhance`: what's left after either reveal method has a thin,
high-frequency vertical-grain residue — leftover dot/compression grain —
while letters are large, low-frequency blobs. A blur that's wider
horizontally than vertically kills that grain while barely softening the
letters, followed by a percentile contrast stretch. (CLAHE was tried and
rejected here — its local tiling re-amplifies exactly the fine-grained
residual grain the blur just suppressed, producing blotchy artifacts
instead of a cleaner image.) Low-confidence regions (see methods 3 and 5
above) are excluded from both the blur and the percentile stretch via a
normalized convolution, rather than being allowed to skew them, and are
rendered as flat mid-gray in the output.

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
