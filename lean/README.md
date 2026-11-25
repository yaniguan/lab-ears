# `LabEarsLean` -- Formal L2 Voice-Privacy, Nyquist & Cross-Modal Acoustic Interlock Proofs

`LabEarsLean` (`leanprover/lean4:v4.36.0`, zero external dependencies) machine-checks the privacy, Nyquist realizability, finite-window STFT skirt margin, outward rounding, and cross-modal fault-detection invariants of `lab-ears`:

- **Voice-Privacy, Nyquist & Finite-Window Skirt Gate (`checkLabEars`, `checkLabEars_strips_speech`, `checkLabEars_nyquist_and_skirt_bounds`)**: Enforces `rawAudioSaved = false`, `sampleRateHz >= 8000`, non-empty `retainedBands`, Nyquist realizability (`2 * hiHz <= sampleRateHz`), calibrated anomaly thresholds, and zero overlap with the `(300, 3400) Hz` human speech band across every retained mel/FFT filter broadened by `windowSkirtHz` (`bandWithSkirt_disjoint_speech_margin`).
- **Conservative Outward Frequency Rounding (`outwardRoundBand`, `outwardRoundBand_contains_scaled`, `outwardRoundBand_disjoint_scaled_speech`)**: Rounds continuous/scaled filter edges `[loScaled / scale, hiScaled / scale]` outward to `[floor(lo), ceil(hi)]` so integer Hz certificates never under-approximate filter support near `300 Hz` or `3400 Hz`.
- **Cross-Modal Cavitation & Stepper-Stall Interlock (`checkCrossModalInterlock`, `crossModalInterlock_nyquist_and_dual_bands`)**: Enforces `sampleRateHz >= 16000` and dual sub-speech (`<= 300 Hz`) + high-frequency (`>= 3400 Hz`) band coverage, and proves that any high-band (`3.6-7.8 kHz`) dry-aspiration cavitation excess (`crossModalInterlock_halts_dry_aspiration`) or low-band (`60-260 Hz`) stepper-stall harmonic excess (`crossModalInterlock_halts_stepper_stall`) during an active pipetting step deterministically asserts `haltTriggered = true` while preserving voice privacy (`crossModalInterlock_preserves_voice_privacy`).

## Build

```bash
cd lean
lake build
```
