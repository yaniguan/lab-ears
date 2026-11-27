# `LabEarsLean` -- Formal L2 Voice-Privacy, Nyquist & Cross-Modal Acoustic Interlock Proofs

`LabEarsLean` (`leanprover/lean4:v4.36.0`, zero external dependencies) machine-checks the privacy, Nyquist realizability, finite-window STFT skirt margin, outward rounding, and cross-modal fault-detection invariants of `lab-ears`:

- **Voice-Privacy, Nyquist & Finite-Window Skirt Gate (`checkLabEars`, `checkLabEars_strips_speech`, `checkLabEars_nyquist_and_skirt_bounds`)**: Enforces `rawAudioSaved = false`, `sampleRateHz >= 8000`, non-empty `retainedBands`, Nyquist realizability (`2 * hiHz <= sampleRateHz`), calibrated anomaly thresholds, and zero overlap with the `(300, 3400) Hz` human speech band across every retained mel/FFT filter broadened by `windowSkirtHz` (`bandWithSkirt_disjoint_speech_margin`).
- **Conservative Outward Frequency Rounding (`outwardRoundBand`, `outwardRoundBand_contains_scaled`, `outwardRoundBand_disjoint_scaled_speech`)**: Rounds continuous/scaled filter edges `[loScaled / scale, hiScaled / scale]` outward to `[floor(lo), ceil(hi)]` so integer Hz certificates never under-approximate filter support near `300 Hz` or `3400 Hz`.
- **Cross-Modal Cavitation & Stepper-Stall Interlock (`checkCrossModalInterlock`, `crossModalInterlock_nyquist_and_dual_bands`)**: Enforces `sampleRateHz >= 16000` and dual sub-speech (`<= 300 Hz`) + high-frequency (`>= 3400 Hz`) band coverage, and proves that any high-band (`3.6-7.8 kHz`) dry-aspiration cavitation excess (`crossModalInterlock_halts_dry_aspiration`) or low-band (`60-260 Hz`) stepper-stall harmonic excess (`crossModalInterlock_halts_stepper_stall`) during an active pipetting step deterministically asserts `haltTriggered = true` while preserving voice privacy (`crossModalInterlock_preserves_voice_privacy`).

## Empirical Speech-Realism Scope Note (`community_extensions/cross_modal_interlock.py`)

The Lean theorems prove deterministic decision-logic soundness and `[300, 3400] Hz` filter disjointness; they do not claim human speech has zero acoustic energy outside `[300, 3400] Hz`. In particular, the `1.0` fault F1 (`0.0` false-alarm rate) reported by the legacy synthetic benchmark held **only** for pure tones confined inside `[380, 3150] Hz`, until real-speech + physical instrument recordings are evaluated.

When evaluated on the 120-clip 16 kHz pipetting benchmark against a broadband speech synthesizer (`F0 100-220 Hz` glottal pulse train with `1/h` harmonics to `7.8 kHz` plus `4-7.8 kHz` sibilant/fricative bursts, `52` speech-overlapped fault-free clips including `44` active transfers):

| Speech Level | Reference | `bandstop_only` False-Halt Rate (All / Active Speech) | `bandstop_only` Fault F1 | `harmonic_onset_aware` False-Halt Rate (All / Active Speech) | `harmonic_onset_aware` Fault F1 |
|---:|---|---:|---:|---:|---:|
| `0 dB` | vs clip RMS | `0.6731` (`0.7955`) | `0.7328` | `0.0000` (`0.0000`) | `0.9787` |
| `-10 dB` | vs clip RMS | `0.0000` (`0.0000`) | `1.0000` | `0.0000` (`0.0000`) | `1.0000` |
| `-20 dB` | vs clip RMS | `0.0000` (`0.0000`) | `1.0000` | `0.0000` (`0.0000`) | `1.0000` |
| `-30 dB` | vs clip RMS | `0.0000` (`0.0000`) | `1.0000` | `0.0000` (`0.0000`) | `1.0000` |
| `0 dB` | vs room-conversation RMS (`+17.6 dB`) | `0.8462` (`1.0000`) | `0.6857` | `0.4808` (`0.5682`) | `0.6727` |
| `-10 dB` | vs room-conversation RMS (`+7.6 dB`) | `0.8462` (`1.0000`) | `0.6857` | `0.3846` (`0.4545`) | `0.7748` |
| `-20 dB` | vs room-conversation RMS (`-2.4 dB`) | `0.3654` (`0.4318`) | `0.8348` | `0.0000` (`0.0000`) | `1.0000` |
| `-30 dB` | vs room-conversation RMS (`-12.4 dB`) | `0.0000` (`0.0000`) | `1.0000` | `0.0000` (`0.0000`) | `1.0000` |

Real 16 kHz speech WAVs (for example local LibriSpeech `CC-BY-4.0`) can be mixed directly via `run_benchmark(speech_dir=...)`.

## Build

```bash
cd lean
lake build
```
