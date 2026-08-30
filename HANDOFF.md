# Handoff

Notes for whoever picks this up, including me in three months.

## Where it stands

Everything is implemented and runs on GPU. Nothing has touched real TITAN data.
I verified the full pipeline runs end-to-end on synthetic data; a real training
run against the actual TITAN dataset is the next step once access is granted.

Read `README.md` first for how to run it, `END_GOAL.md` for what finished looks
like, `PROGRESS.md` for the checklist.

## The one blocker: dataset access

TITAN is not a public download. To get it:

1. Go to <https://usa.honda-ri.com/titan>.
2. Fill in the access request form. It asks for your name, affiliation, and
   what you plan to use the data for. Academic and research use is the expected
   case.
3. Honda Research Institute reviews the request. If approved they email
   download links.

Check the page for the current process; it is theirs and can change. Budget
real calendar time for the review — this is not an instant download, and it is
the reason this repo has no results rather than any technical obstacle.

Once it arrives, extract into the layout in the README's "Where to put the
data" section. `data.root` defaults to `data/titan`.

## What to do first, in order

1. **Get the loader to produce a batch.** `src/titan/data/titan.py` was written
   from the published schema and has never seen a real file. Two things will
   probably need fixing: the annotation CSV column names (resolved by alias in
   `data/schema.py`, so add aliases rather than renaming), and the frame
   filename zero-padding (`data/video.py::_find_frame` tries several widths).
   Expect this to be a debugging session.

2. **Run the constant-velocity baseline on real data before training anything.**
   Every epoch prints it, but you can get it out of an untrained model with
   `titan.cli eval`. The paper reports 102.47 px FDE. If yours is not in that
   neighbourhood, something in the loading or the eval protocol is wrong —
   frame rate subsampling, box coordinate convention, the pixel scale — and
   every number after it will be wrong too. This is the cheapest possible check
   and it needs no training at all. Do not skip it.

3. **Then train `EP+IP+AP`.** Then the full ablation.

4. **Check the ordering before the magnitudes.** The paper's claim is that each
   prior adds something. If you get the ordering right but weaker absolute
   numbers, that is a partial reproduction and should be written up as exactly
   that. Do not round it up into a claim of reproduction.

## Design decisions worth defending

### r3d_18 instead of I3D — the substitution that matters most

The paper finetunes single-stream I3D and a 3D ResNet, both pretrained on
Kinetics-600. This repo uses `torchvision.models.video.r3d_18`, a 3D ResNet-18
pretrained on Kinetics-400.

Why: Kinetics-600 I3D weights are not distributed with torchvision and there is
no first-party PyTorch I3D that drops in cleanly. Pulling in a third-party I3D
port plus converted weights would have added a dependency I could not verify
and an unpinnable weight file, for a repo that cannot yet run on real data
anyway. `r3d_18` is a genuine 3D-conv video backbone available from a first-
party source, so the pipeline is real even though the backbone is smaller.

What it costs: this is a substitution for the 3D-ResNet arm the paper also
reports, **not** a reimplementation of I3D, and it is a smaller network on a
smaller pretraining set. Expect the AP rows to gain less than the paper's do.
If the ablation ordering reproduces but AP contributes less than expected, this
is the first thing to change.

`model.action_backbone` also accepts `mc3_18` and `r2plus1d_18`, so trying the
mixed-convolution and (2+1)D variants is a config change. Wiring in a real I3D
with Kinetics-600 weights is the highest-value faithfulness fix on the list.

Every banner that mentions the backbone prints `r3d_18 (substitute for I3D, see
README)`. The repo never claims I3D. Keep it that way.

### AIM's scoring function is invented

The paper describes the Agent Importance Mechanism as a self-attention-style
weighting, `H~ = w * H` with `w = phi(H)`, learned jointly with future
ego-motion prediction — but does not publish the shape of `phi`.
`models/aim.py` uses a two-layer MLP with a tanh, a mask-aware softmax over
agents actually present in the frame, and a weighted pool.

Two details that are not arbitrary: padded agent slots are masked to `-inf`
before the softmax so they cannot steal probability mass from real agents, and
a frame with zero agents has its weights zeroed rather than left as uniform
garbage. Both are tested.

The weights are returned from `forward` rather than kept internal. Attributing
ego-vehicle risk to individual agents is the interpretability claim the module
exists to support, so it should be reachable.

### Interaction runs after the action feature is concatenated

So a neighbour's action representation is visible to the attention, not just
their trajectory. That is my reading of the paper's ordering and I am not
certain of it. If IP and AP appear to fight each other on real data, try
running interaction on the trajectory encoding alone — it is a small change in
`titan_net.py`.

### The decoder predicts deltas

Box deltas integrated onto the last observed box, rather than absolute pixel
coordinates. Predicting absolute coordinates makes the GRU spend capacity
relearning the identity of the last observed box at every step instead of
learning the dynamics. Standard, but worth knowing it is a choice.

### Boxes normalised, metrics in pixels

The network works on coordinates divided by 1920x1200; every reported metric
converts back to pixels first. That conversion lives in exactly one place
(`data/common.py`) so there is one thing to get wrong instead of many. The
smooth-L1 `beta=0.01` is set so that errors around the paper's target scale
(~20 px, i.e. ~0.01 normalised) sit in the quadratic region while badly tracked
agents fall in the linear region and do not dominate the update.

### FDE uses the last valid step

Not index -1. Tracks drop out mid-horizon, and taking index -1 on a track that
ended early measures the error against padding. `metrics.py::_last_valid_index`
handles it and it is tested directly.

### AMP: bf16 without a scaler, fp16 with one

`device.py::select_precision` picks bf16 when the card supports it. bf16 keeps
fp32's exponent range so gradients cannot underflow and loss scaling is
pointless; fp16 gets a `GradScaler`. Gradients are unscaled before clipping —
clipping scaled gradients would make the threshold meaningless.

ADE/FDE are computed outside autocast, in fp32, deliberately. They are the
reported numbers.

## Gotchas

- **Triton needs `cuda.h`.** An earlier smoke test died on the first backward
  pass with `fatal error: cuda.h: No such file or directory`. Torch routes some
  backward ops through Triton, which JIT-compiles a helper that includes that
  header, and the CUDA *runtime* packages do not ship it. Fixed by
  `cuda-cudart-dev` in `pixi.toml`. On the pip path you may need the CUDA
  toolkit headers.

- **`checkpoints/` and `data/` are gitignored.** So are `*.pt`. Only
  `artifacts/` is committed, and it holds captured stdout, not weights.

- **Turning off AP turns off image loading entirely.** In
  `engine.build_dataset`. Decoding PNG crops dominates load time and nothing
  downstream consumes the tubes when AP is off, so the non-AP ablation rows run
  much faster. Do not read that speed difference as a bug.

- **`strict_split_sizes` will stop you.** The loader refuses to start if the
  split files do not hold exactly 400/200/100 clips, so a partial download
  fails loudly instead of quietly training on less data. Set it false only if
  you are deliberately using a subset — and then say so in any write-up.

- **`--compile` is off by default** and is not well tested here. It should help
  a long training run; it is pure overhead on a smoke test. If you enable it,
  note that `torch.compile` returns a wrapper that does not forward attribute
  access, which is why `engine.train` keeps a handle on the uncompiled module
  for `state_dict` and `action_log_var`.

- **The synthetic data is not a weak dataset, it is a fake one.** Random walks
  with noise, and the video tubes are literally Gaussian noise. Its only job is
  to prove tensors flow. Any FDE from it is a fact about random walks.

## Honesty rules for this repo

The paper is published, the dataset is real, and anyone can check the numbers.
So:

- The paper's numbers live in `src/titan/paper.py` and are never printed
  without a header saying the authors measured them, not this code.
- Synthetic runs print a banner at the start, tag every summary line
  `[synthetic]`, print a warning at the end, and set `"synthetic": true` in
  `history.json`. Four places, on purpose.
- The synthetic summary table and the paper's table are printed as two separate
  blocks and are never merged into one.
- If a run did not happen, its number does not get written down.
