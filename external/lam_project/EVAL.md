# LAM checkpoint evaluation

## Baseline suite (v2)

The checkpoint-selection baseline evaluates the released 400k-step LAM and
retained continued checkpoints on one immutable, accepted-only,
source-balanced set of 5,000 frame pairs from a pinned Lance version.

In addition to image reconstruction, v2 aligns each camera pair with the
canonical timestamped 14-D bilateral robot action stored in Lance:

- left/right end-effector translation (XYZ);
- left/right relative rotation vectors (XYZ);
- left/right normalized gripper deltas.

Smoothed global deltas are preferred, with explicit global/local fallbacks.
Each target dimension is used only when every action step spanning the frame
pair is finite and valid. The suite reports:

- MSE, PSNR, SSIM, and LPIPS, micro-, macro-, and per-source;
- zero- and shuffled-latent reconstruction ablations;
- source-stratified, episode-clustered 95% bootstrap intervals;
- latent variance, posterior scale, effective rank, and participation ratio;
- episode-split optical-flow and canonical robot-action linear probes;
- leave-one-source-out canonical-action transfer;
- source and camera nearest-centroid leakage probes;
- action-label coverage, per-sample JSONL, montages, and summary CSV.

Run a checkpoint with:

```bash
python evaluate_lam.py \
  --checkpoint /checkpoints/model.ckpt \
  --dataset-kind lance \
  --dataset-path /data/episodes.lance \
  --dataset-version 32949 \
  --manifest /eval/umi-manifest.json \
  --output-dir /eval/umi \
  --num-samples 5000
```

Completed checkpoints are skipped. Outputs are written to:

Use downstream action quality as the primary selection signal, nuisance
leakage and latent health as gates, and reconstruction as a sanity check.

## External Pantheon teleop suite

The Pantheon teleop suite is a clean external-domain baseline for UMI-only
continuation. It reads a prepared Pantheon teleop manifest and filters strictly
to `disposition == "ACCEPT"`.

The source manifest supplies a frozen split: 141 accepted episodes fit the
linear probes and four accepted validation episodes score them. Validation
reconstruction is also computed only on those four episodes. The suite samples
four windows per probe-train episode and 64 windows per validation episode,
while confidence intervals resample whole episodes rather than treating the
256 validation windows as independent observations.

The 14-D target is the change in issued absolute YAM command between video
endpoints: six joints plus gripper for the left arm, followed by the same seven
dimensions for the right arm. Results must be read with their four-episode
confidence intervals; this is a clean external check, but still a small and
single-task test set.

Run it with:

```bash
python evaluate_lam.py \
  --checkpoint /checkpoints/model.ckpt \
  --dataset-kind pantheon-teleop \
  --dataset-path /data/pantheon-teleop \
  --manifest /eval/teleop-manifest.json \
  --output-dir /eval/teleop
```

## Fast trend suite (v1)

`evaluate_lam.py` scores every checkpoint against one immutable, accepted-only
benchmark drawn from Lance version `32949`. The manifest stores exact row IDs,
cameras, time gaps, and frame indices, so every checkpoint sees identical input.

The benchmark reports:

- reconstruction MSE, PSNR, SSIM, and LPIPS, both micro-averaged and per source;
- reconstruction with the inferred latent replaced by zero or shuffled across
  the batch, which tests whether the decoder actually uses the latent action;
- latent variance, posterior scale, effective rank, and participation ratio;
- a deterministic linear probe from the latent to optical-flow statistics;
- a nearest-centroid source probe to expose source-specific shortcuts;
- per-sample JSONL, a reconstruction montage, and a cross-checkpoint CSV.

The evaluator skips a completed checkpoint unless `--overwrite` is passed, so
the same command can safely be reused as new retained checkpoints appear.

## Leakage status

If training samples all accepted rows from the same pinned Lance version, the
fixed benchmark is suitable for comparing checkpoints but is not a
leakage-free generalization estimate. Label such results as checkpoint trends.

For a fresh leakage-free run, use
`config/lam_umi_accepted_continued_700m_b160_holdout.yaml`. It applies the same
per-source, deterministic one-percent episode split used by the evaluator and
excludes those row IDs from training.
