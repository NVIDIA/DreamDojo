# Accepted-only LAM continuation and evaluation

This subproject extends DreamDojo's latent action model (LAM) training path
with accepted-only UMI data, distributed continued pretraining, bounded
checkpoint retention, and action-centered checkpoint evaluation.

> [!IMPORTANT]
> The reference experiment improved the in-domain UMI trend benchmark, but it
> did **not** improve clean external teleoperation transfer. Training was
> stopped at 10,000 continuation steps and the 10k checkpoint was retained.

## Executive summary

The experiment continued the released 400k-step DreamDojo LAM rather than
training a larger model or starting from random initialization. It used the
same approximately 710M-parameter architecture and the same main optimizer and
frame-pair settings, while replacing the example video-folder input with a
pinned, accepted-only Lance snapshot containing 351,169 eligible UMI episodes.

Three checkpoints were compared with the released base model:

- 500 continuation steps;
- 2,000 continuation steps;
- 10,000 continuation steps.

At 10k, canonical-action mean R² on the 5,000-sample UMI trend suite increased
from `0.0448` to `0.0699`, while PSNR increased by `1.18 dB`. However, the gain
over step 2k was only `+0.0064`, below the predeclared `+0.02` continuation
threshold. On the clean external Pantheon teleop suite, action R² remained
negative (`-0.0351`) and reconstruction became substantially worse than the
released base. The 10k checkpoint showed no non-finite metrics or obvious
latent collapse; the result is best understood as in-domain adaptation without
usable cross-domain action transfer.

## What this change adds

| Area | Implementation |
| --- | --- |
| Accepted-only training | Lance-backed sampling with an exact `accepted` and valid capture policy |
| Reproducibility | Pinned Lance dataset version and deterministic episode splits |
| Continued pretraining | Strict initialization from the released `LAM_400k.ckpt` weights |
| Multi-node correctness | Sample-weighted loss scaling for heterogeneous local batches |
| Checkpoint retention | One rolling checkpoint every 500 steps and immutable 10k milestones |
| UMI evaluation | Fixed, source-balanced reconstruction, latent-health, leakage, motion, and 14-D canonical-action metrics |
| External evaluation | Accepted-only Pantheon teleop evaluation with frozen episode-level probe splits |
| Verification | Unit coverage for filtering, manifests, action alignment, statistics, and checkpoint retention |

The exact evaluation commands and output schema are documented in
[EVAL.md](EVAL.md).

## Reference training run

### Model and optimization

| Setting | Value |
| --- | --- |
| Initialization | Released DreamDojo `LAM_400k.ckpt` |
| Architecture | 24 encoder blocks, 24 decoder blocks, width 1024, 16 heads |
| Parameters | Approximately 710M |
| Latent action | 32 dimensions |
| Patch size | 16 |
| Objective | Future-frame MSE + β-weighted latent KL |
| β | `1e-6` |
| Optimizer | AdamW |
| Learning rate | `2.5e-5` |
| Weight decay | `1e-2` |
| Precision | 16-bit mixed precision |
| Gradient clipping | `0.3` |
| Frame-pair gaps | 0.1, 0.2, 0.3, or 0.4 seconds |
| Cameras | Left wrist, right wrist, or overhead |
| Global batch | Exactly 160 samples per optimizer step |
| Compute | 3 nodes × 8 H200 GPUs |

The exact batch of 160 used heterogeneous local batches: 16 ranks processed
seven samples and eight ranks processed six. Their local losses were scaled by
`1.05` and `0.90`, respectively, so DDP produced the same sample-weighted
gradient as a uniform global batch. A launcher reproducing this layout must
apply the six-sample/`0.90` override to the final eight ranks; running the
checked-in seven-sample config unchanged on all 24 ranks would instead produce
a batch of 168.

The stable portion of the run sustained approximately `0.33 steps/s`. Reaching
10k required about 8 hours 21 minutes of active training, or roughly 200 H200
GPU-hours, excluding evaluation and setup.

### Training data

The training dataset was the pinned Lance snapshot at version `32949`, filtered
by the non-configurable policy:

```text
quality.capture_status = 'accepted' AND quality.capture_valid = true
```

That snapshot contained 351,169 eligible episodes across ten source datasets.
Only `left_wrist`, `right_wrist`, and `overhead` embedded videos were eligible.
The loader samples a source, episode, camera, and time gap, decodes the two
frames, and rechecks the accepted-only invariant before returning a sample.

This number should not be presented as a direct “more data than the released
DreamDojo LAM” comparison: the released checkpoint does not provide an
equivalent unique-episode inventory in this repository. It is, however, much
broader than the GR-1-only example configuration originally included here.

### Checkpoint policy

The reference run wrote a recoverable rolling checkpoint every 500 steps,
atomically replacing the previous rolling file. Step-numbered checkpoints are
kept every 10,000 steps, with explicitly requested early checkpoints preserved.
The reference comparison retained steps 500, 2k, and 10k.

The corresponding configuration is
[`config/lam_umi_accepted_continued_700m_b160.yaml`](config/lam_umi_accepted_continued_700m_b160.yaml).
A leakage-free follow-up configuration is provided at
[`config/lam_umi_accepted_continued_700m_b160_holdout.yaml`](config/lam_umi_accepted_continued_700m_b160_holdout.yaml).

## Evaluation design

Checkpoint selection treated action quality as the primary signal,
reconstruction as a sanity check, and latent health plus nuisance leakage as
gates. All checkpoint comparisons used identical frozen samples.

### Accepted-only UMI trend suite

The UMI suite contains 5,000 source-balanced frame pairs from Lance version
`32949`. It reports:

- MSE, PSNR, SSIM, and LPIPS;
- reconstruction with zero and shuffled latent actions;
- latent variance, posterior scale, effective rank, and participation ratio;
- episode-split optical-flow and 14-D canonical robot-action linear probes;
- leave-one-source-out action transfer;
- source and camera nearest-centroid leakage probes;
- episode-clustered, source-stratified bootstrap intervals.

The canonical 14-D target is bilateral end-effector translation, relative
rotation, and normalized gripper change. Because the reference training run
sampled the full accepted pool from the same Lance snapshot, this suite overlaps
the training population. Its results are valid for checkpoint trends, not as a
leakage-free generalization estimate.

### Accepted-only Pantheon teleop suite

The external suite was prepared from 642 teleoperation episodes: 145 were
marked `ACCEPT` and 497 were excluded as `REVIEW`. The frozen split used 141
accepted episodes to fit probes and four accepted episodes for validation.
It evaluated 256 validation windows and 820 total probe samples.

The action target is the change in a 14-D issued absolute YAM command between
the two video endpoints: six joints and one gripper value for each arm.
Bootstrap resampling treats episodes—not individual windows—as the independent
unit.

This is a clean external-domain check, but it is small: all four validation
episodes perform “Place the duck in the box” and cover only two robots. Its
absolute values should therefore be read together with the episode-level
uncertainty and confirmed on a larger task-diverse holdout.

## Results

Higher PSNR, SSIM, motion R², and action R² are better. Lower MSE and LPIPS are
better. A negative action R² is worse than predicting the probe-training target
mean.

### UMI trend results

| Checkpoint | MSE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Motion R² ↑ | Action R² ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Released base 400k | 0.004061 | 28.1543 | 0.8652 | 0.2276 | 0.5715 | 0.0448 |
| Continued 500 | 0.003242 | 28.5403 | 0.8764 | 0.2256 | 0.5774 | 0.0576 |
| Continued 2k | 0.002985 | 29.1140 | 0.8819 | 0.2180 | 0.5594 | 0.0635 |
| **Continued 10k** | **0.002741** | **29.3367** | **0.8861** | **0.2110** | 0.5409 | **0.0699** |

From the released base to 10k, MSE fell by 32.5%, PSNR rose by 1.1824 dB,
SSIM rose by 0.0208, LPIPS fell by 0.0165, and action R² rose by 0.0251.
Most of the action gain arrived early: step 2k to step 10k added only 0.0064.
The optical-flow probe declined over the same interval.

### UMI latent and leakage diagnostics

| Checkpoint | Zero-latent MSE | Shuffled-latent MSE | Effective rank ↑ | Cross-source action R² ↑ | Source probe acc. ↓ | Camera probe acc. ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Released base 400k | 0.01856 | 0.02698 | 27.00 | -6.5936 | 0.3034 | 0.5427 |
| Continued 500 | 0.01868 | 0.02749 | 27.36 | -6.4990 | 0.2730 | 0.5326 |
| Continued 2k | 0.01878 | 0.02803 | 27.55 | -7.0191 | 0.2921 | 0.5146 |
| Continued 10k | 0.01876 | 0.02784 | 27.74 | -8.0861 | 0.2708 | 0.5292 |

The full reconstruction remains far better than either latent ablation, and
effective rank remains high, so the 10k model did not exhibit obvious latent
collapse. However, leave-one-source-out action R² is strongly negative and
worsens after 500 steps. The positive pooled action probe is therefore not
evidence of source-independent action semantics.

For context, the released base reconstruction bootstrap medians and 95%
intervals were PSNR `28.1595 [27.9504, 28.3800]`, SSIM
`0.8653 [0.8621, 0.8685]`, LPIPS `0.2276 [0.2239, 0.2312]`, and MSE
`0.004056 [0.003867, 0.004249]` over 1,652 unique episodes.

### External Pantheon teleop results

| Checkpoint | MSE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Motion R² ↑ | Action R² ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Released base 400k** | **0.001211** | **31.3321** | **0.9371** | **0.0867** | 0.6719 | -0.0576 |
| Continued 500 | 0.001595 | 29.8832 | 0.9176 | 0.1277 | 0.6912 | -0.0480 |
| Continued 2k | 0.001734 | 29.8317 | 0.9210 | 0.1304 | 0.6892 | -0.0498 |
| Continued 10k | 0.002105 | 28.9491 | 0.9126 | 0.1504 | **0.7076** | **-0.0351** |

The 10k action probe is 0.0225 above the released base, but it remains negative
and is only 0.0148 above step 2k. Meanwhile, relative to the base, MSE rises by
73.8%, PSNR falls by 2.3830 dB, SSIM falls by 0.0245, and LPIPS worsens by
73.5%. Continued UMI-only training therefore does not produce usable external
teleop action transfer under this evaluation.

### External latent diagnostics

| Checkpoint | Zero-latent MSE | Shuffled-latent MSE | Effective rank ↑ | Cross-source action R² ↑ | Source probe acc. ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Released base 400k | 0.007638 | 0.009419 | 16.06 | -0.9464 | 0.3750 |
| Continued 500 | 0.007829 | 0.009438 | 15.94 | -0.9250 | 0.3594 |
| Continued 2k | 0.007911 | 0.009605 | 16.28 | -0.9391 | 0.3867 |
| Continued 10k | 0.008574 | 0.009533 | 15.69 | -1.4011 | 0.4141 |

The released-base episode bootstrap medians and 95% intervals were PSNR
`31.3321 [30.6385, 32.7086]`, SSIM `0.9371 [0.9331, 0.9424]`, LPIPS
`0.0867 [0.0824, 0.0908]`, and MSE
`0.001211 [0.000885, 0.001438]`. These intervals reflect only four validation
episodes and should not be mistaken for broad task coverage.

## Stop decision

The predeclared 10k gate allowed training to continue only if the checkpoint
was finite, showed no clear latent collapse, and met either condition:

1. UMI canonical-action mean R² was at least `0.02` above both the released
   base and step 2k; or
2. external teleop action mean R² became positive and was at least `0.02`
   above step 2k.

For UMI, 10k needed at least `0.0835` and achieved `0.0699`. For teleop, it
needed to become positive and achieved `-0.0351`. The run failed both action
gates, so it was canceled after the permanent 10k checkpoint was verified and
both evaluations completed. The 10k checkpoint and evaluation outputs were
retained.

## Recommendation

- Use the released 400k checkpoint for external teleop or general-domain work.
- Use the 10k checkpoint only when selecting specifically for the overlapping
  accepted-UMI domain, and label those results as trend-only.
- Do not resume the same UMI-only configuration expecting cross-domain action
  transfer; the marginal action gain had already flattened by 2k–10k.
- For the next run, exclude a deterministic UMI episode holdout from training,
  mix accepted teleop data into training, balance sources or domains explicitly,
  and evaluate on a larger task- and robot-diverse external split.
- Consider an action-aware auxiliary objective or alignment loss if the goal is
  source-independent control semantics rather than reconstruction alone.

## Running the code

The original single-node and multi-node entry points remain:

```bash
cd external/lam_project
bash train.sh
```

```bash
cd external/lam_project
bash launch.sh 0
```

For accepted-only continuation, pass the relevant configuration through the
same Lightning entry point used by those launchers. Before a long run, use
[`config/lam_umi_accepted_3node_canary.yaml`](config/lam_umi_accepted_3node_canary.yaml)
to validate dataset access, distributed startup, and checkpoint writes.

Run the focused test suite with:

```bash
cd external/lam_project
python -m unittest discover -s tests -v
```

See [EVAL.md](EVAL.md) for evaluator commands, metric definitions, manifest
requirements, and leakage guidance.
