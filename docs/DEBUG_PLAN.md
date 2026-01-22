# Debug Plan: Live BP Collapse to Mean

## Current Problem
- Live BP outputs stay near label mean (~143/66) while heart rate varies.
- Raw model outputs are ~0 in normalized space → inverse scaling returns label mean.
- Webcam signal SNR was low (e.g., -3.5 dB) and preprocessing differs from training.

## Likely Causes
- Input distribution mismatch: training used dataset-level z-score only; realtime adds detrend + adaptive filtering + smoothing + z-score.
- Low signal quality (motion/lighting/ROI) leading to noisy POS waveform.
- Possible scaler mismatch or missing stats.

## Debug Plan (Ordered)
1) **Reproduce on test split**
   - Run `python -m training.train_transformer --data-dir data --epochs 0` (or a small eval snippet) to ensure the trained model predicts correctly on `rppg_test.h5` with inverse scaling. If it collapses there, model/weights are broken; otherwise it’s a runtime input issue.

2) **Align preprocessing**
   - Add a flag in `realtime/integrated_pipeline.py` to optionally skip detrend/adaptive/smoothing and only z-score with dataset scalers; compare outputs to current path.
   - If skipping extra steps restores variability, retrain/fine-tune using the realtime preprocessing chain.

3) **Check scaler usage**
   - Verify `data/rppg_info.txt` is loaded (length 875 for signal_mean/scale; label_mean/scale populated).
   - Log a sample normalized signal snippet and stats (mean/std) before model inference to confirm non-zero variance.

4) **Measure signal quality**
   - Use `tests/debug_realtime_test.py` to log SNR, quality metrics, and raw model outputs side-by-side; target SNR > 0 dB.
   - Improve acquisition: stable head, strong ambient light, larger face ROI; avoid compression artifacts.

5) **A/B model inputs**
   - Feed a known-good test window from `rppg_test.h5` through the realtime path (bypassing camera) to ensure the pipeline yields expected BP. This isolates camera/ROI issues from preprocessing.

6) **Post-processing sanity**
   - Temporarily bypass `BPStabilizer` to ensure variability isn’t being flattened post-inference.

7) **Performance/logging**
   - Add structured logging (CSV/JSON) of `[timestamp, hr, snr, sbp_raw_model, dbp_raw_model, sbp_raw, dbp_raw, quality]` for short sessions to analyze correlations.

## Fix Path
- If test-set eval is good but live is flat: focus on preprocessing alignment and SNR improvements; retrain/fine-tune with realtime preprocessing.
- If test-set eval is also flat: retrain or check weights/label scaling.

## Next Actions
- [ ] Run test-set evaluation to confirm model health.
- [ ] Add toggle to skip extra preprocessing in realtime and compare.
- [ ] Log normalized signal stats and raw model outputs during live run.
- [ ] Capture a good rPPG window from test data and run through realtime pipeline (no camera) to compare.
- [ ] Decide on retrain/fine-tune with realtime preprocessing if mismatch persists.
