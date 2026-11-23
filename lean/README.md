# `LabEarsLean` -- Formal L2 Voice-Privacy & Cross-Modal Acoustic Interlock Proofs

`LabEarsLean` (`leanprover/lean4:v4.36.0`, zero external dependencies) machine-checks the privacy and cross-modal fault-detection invariants of `lab-ears`:

- **Voice-Privacy Gate (`checkLabEars`, `checkLabEars_strips_speech`)**: Enforces `rawAudioSaved = false`, `sampleRateHz >= 8000`, calibrated anomaly thresholds, and zero interior overlap with the `(300, 3400) Hz` human speech band across every retained mel/FFT filter skirt (`bandOutsideSpeech_disjoint_interior_speech`).
- **Cross-Modal Cavitation & Stepper-Stall Interlock (`checkCrossModalInterlock`)**: Proves that any high-band (`3.6-7.8 kHz`) dry-aspiration cavitation excess (`crossModalInterlock_halts_dry_aspiration`) or low-band (`60-260 Hz`) stepper-stall harmonic excess (`crossModalInterlock_halts_stepper_stall`) during an active pipetting step deterministically asserts `haltTriggered = true` while preserving voice privacy (`crossModalInterlock_preserves_voice_privacy`).

## Build

```bash
cd lean
lake build
```
