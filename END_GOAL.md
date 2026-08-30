# What "done" looks like

The goal is a defensible reproduction attempt of TITAN (Malla, Dariush & Choi,
CVPR 2020), where every number I report is a number I measured, and every gap
between my numbers and the paper's is explained rather than hidden.

Not "get 19.5 px FDE". That is the paper's number. Mine will be whatever it is.

## Hard prerequisite: real TITAN dataset access

**Nothing below the smoke test can happen without it.** TITAN is access-gated:
request through <https://usa.honda-ri.com/titan>, HRI reviews, and if approved
they send download links. I have not been granted access. Until that lands,
this repo can only demonstrate that the pipeline runs, which is exactly what it
currently claims and nothing more.

There is no workaround. There is no substitute dataset that would make a
reproduction claim honest. If access is refused, the correct outcome is a repo
that says "implemented, never validated against real data" — not one that
quietly reports synthetic numbers.

## Definition of done

### 1. It runs on anyone's machine — done

- [x] Clone, install (pixi or venv+pip), and get a working GPU environment.
- [x] `scripts/check_gpu.py` reports device, VRAM, precision support, and a
      working matmul.
- [x] Tests pass.
- [x] A synthetic end-to-end smoke test exercises all seven ablation paths.
- [x] Both install paths verified on real hardware, not assumed.

### 2. The implementation is faithful — mostly done

- [x] EP / IP / AP on independent switches, so all seven of the paper's
      ablation rows are runnable.
- [x] Agent Importance Mechanism, with per-agent weights exposed rather than
      hidden, trained jointly with ego-motion prediction.
- [x] Action recognition as five parallel multi-class heads with the paper's
      uncertainty weighting, not one flat softmax.
- [x] ADE / FDE / FIOU in pixels at 1920x1200, computed in fp32.
- [x] Constant-velocity baseline, which is the one row of the paper's table
      computable with no training at all.
- [ ] A real I3D action branch. Currently `r3d_18` stands in — a substitution
      for the 3D-ResNet arm the paper also reports, not for I3D. Documented in
      the README; still a real gap in faithfulness.
- [ ] The data loader validated against the actual tarballs. It is written from
      the published schema and has never seen real TITAN files.

### 3. A real result exists — blocked on access

- [ ] Dataset access granted and data extracted.
- [ ] Constant-velocity baseline computed on real TITAN. This is the gate: if
      it does not land near the paper's 102.5 px FDE, the loader or the eval
      protocol is wrong and every later number is meaningless. Debug here
      before training anything.
- [ ] `EP+IP+AP` trained to convergence on the real train split, evaluated on
      the real test split.
- [ ] The full seven-row ablation run on real data.
- [ ] Peak VRAM, samples/sec and wall clock recorded for each run, so the cost
      of the result is known.

### 4. The write-up is honest — the part that actually matters

- [ ] A results table with my numbers and the paper's side by side, clearly
      labelled as two different things.
- [ ] The ablation ordering checked: does each added prior improve FDE? Getting
      the ordering right with worse absolute numbers is a partial reproduction
      and should be described exactly that way.
- [ ] Every gap explained. Candidates, in the order I would suspect them:
      backbone substitution (r3d_18 vs I3D, Kinetics-400 vs 600), unpublished
      hyperparameters, my reading of the AIM scoring function, my choice to run
      interaction after the action feature is concatenated, and train/val/test
      protocol details the paper does not fully specify.
- [ ] Anything I could not reproduce, stated as not reproduced.

## The line I will not cross

If a run does not happen, its number does not get written down. If a number
comes from synthetic data, it says so on the same line. If the reproduction
falls short, the write-up says by how much and why.

This paper is published, the dataset is real, and the numbers are checkable by
anyone who cares to. A fabricated result here would be both dishonest and
trivially caught, which is the correct set of incentives for a reproduction.
