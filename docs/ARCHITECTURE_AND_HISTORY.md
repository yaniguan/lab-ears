# `lab-ears` -- Architecture and History

> **Auto-Lab Ecosystem**: [`auto-lab`](https://github.com/yaniguan/auto-lab) · [Project Studio](https://yaniguan.github.io/auto-lab/)  
> **Layer**: `L2 · Auditory & Protocol Sensory Cortex`  
> **License**: MIT (repository code; external datasets retain their respective licenses)

---

## 1. Object

`lab-ears` is the **Auditory & Protocol Sensory Cortex (`L2`)** of the [Robot AI Auto-Lab](https://github.com/yaniguan/auto-lab). Within the `L0`–`L6` Brain/Body architecture, `L2` is responsible for sensory ingestion across two complementary channels before any downstream reasoning, verification, or robotic actuation takes place:

1. **Written & Spoken Protocol Comprehension (`demo/parse_protocol.py` & `protocol/learned_tagger.py`)**: Extracting structured, unit-normalized experimental parameters (`Amount`, `Temperature`, `Time`, `Speed`, and optional `Concentration` character spans) from free-form wet-lab protocol prose via both a deterministic stdlib regular-expression baseline and a zero-leakage Passive-Aggressive (`PA-II`) linear-chain 17-state `BIOES` sequence tagger.
2. **Unsupervised Machine Acoustic & Cross-Modal Interlock Monitoring (`acoustic/` & `community_extensions/cross_modal_interlock.py`)**: Learning a one-class representation of normal benchtop instrument sound per machine without anomalous training examples, calibrating causal streaming chunk thresholds on matching window durations, monitoring step-synchronized pipetting transfers for cavitation and stepper stall, and emitting validated `lab-ears/audio-event/v1` ledger records while excising `[300, 3400] Hz` human speech frequencies and persisting zero raw waveforms.

### Inputs and Outputs

| Track | Module | Input | Output |
|---|---|---|---|
| Rule-based protocol extraction | `demo/parse_protocol.py` | Plain-text protocol files (`*.txt`) and optional BRAT standoff files (`*.ann`) | `(label, start_char, end_char)` spans (`find_spans`), canonical `ParsedQuantity` intervals (`parse_quantities`), and exact / 1-to-1 bipartite overlap `P/R/F1` |
| Learned sequence span tagging | `protocol/learned_tagger.py` | BRAT `train`/`dev`/`test` splits or normalized `protocols.jsonl` corpora | Constrained 17-state `BIOES` Viterbi spans, canonical `ParsedQuantity` records, non-pickle `.json.gz` bundles, and SHA-256 split-hygiene reports (`results/wlp_learned_vs_rules.json`) |
| Acoustic anomaly & streaming monitoring | `acoustic/` | `16 kHz` (or `>= 16 kHz` downsampled) WAV clips or causal PCM ring-buffer chunks | `(n_frames, n_mels)` float32 speech-stripped log-mel spectrograms, window-matched conformal thresholds, `results/dcase2023_cpu.json`, and `lab-ears/audio-event/v1` JSONL events |
| Cross-modal pipetting interlock | `community_extensions/cross_modal_interlock.py` | Step-indexed `PipettingCommandWindow` + benchtop `>= 16 kHz` audio | `InterlockDecision` (`bandstop_only` or `harmonic_onset_aware` comb-valley + onset speech rejection) |

---

## 2. Data

### 2.1 Active Benchmark Datasets

1. **Wet Lab Protocols Corpus (WLP)** -- [`chaitanya2334/WLP-Dataset`](https://github.com/chaitanya2334/WLP-Dataset) (Kulkarni et al., NAACL 2018, `arXiv:1805.00195`, **MIT License**, pinned commit `1f51ae6e620a5aa3590acf725713f430783f56d7`):
   - `622` wet-lab biology and chemistry protocols (`374` `train`, `123` `dev`, `125` `test`) in BRAT standoff format (`*.txt` + `*.ann`), with semicolon-separated discontinuous `T*` entities split into contiguous segments (`include_discontinuous=True` default):
     - **`train` (`374` protocols, `1` discontinuous entity)**: `2,814` `Amount`, `1,400` `Temperature`, `2,100` `Time`, `591` `Speed` (`6,905` gold spans).
     - **`dev` (`123` protocols, `1` discontinuous entity)**: `977` `Amount`, `460` `Temperature`, `679` `Time`, `215` `Speed` (`2,331` gold spans).
     - **`test` (`125` protocols, `2` discontinuous entities)**: `1,013` `Amount`, `509` `Temperature`, `812` `Time`, `155` `Speed` (`2,489` gold spans).
     - **Pooled (`622` protocols)**: `4,804` `Amount`, `2,369` `Temperature`, `3,591` `Time`, `961` `Speed` (`11,725` gold spans).
   - Upstream corpus note: SHA-256 content auditing identifies `2` `dev` files (`protocol_31`, `protocol_61`) and `4` `test` files (`protocol_291`, `protocol_468`, `protocol_483`, `protocol_571`) whose `.txt` prose duplicates a `train` protocol; `protocol/learned_tagger.py` reports both full-split and unseen-SHA256 metrics.

2. **DCASE 2023 Challenge Task 2 Development Set** -- [Zenodo Record 7882613](https://zenodo.org/records/7882613) (**CC BY 4.0**):
   - Evaluated across `fan`, `valve`, and `bearing` (`1,000` normal `10 s` `16 kHz` training clips = `990` source + `10` target domain; `200` `10 s` test clips = `50` source normal, `50` source anomaly, `50` target normal, `50` target anomaly).

3. **Open Sensory Dataset Registry (`acoustic/open_datasets.py`)**:
   - 12 typed `OpenSensoryDataset` entries (`dcase2024_task2_dev`, `dcase2025_task2_dev`, `dcase2026_task2_dev`, `imad_ds_2024`, `mimii`, `mimii_dg`, `toyadmos2`, `lab_bench_protocolqa`, `chembench`, `bioprobench`, `bioplanner_bioprot`, `protoact_biop2e`) with MD5 verification, Zip-Slip guards, and fail-closed `commercial_use_ok` gates.

### 2.2 Event Schema (`lab-ears/audio-event/v1`)

Defined in `acoustic/record.py` (`AudioEvent` dataclass):
- Required fields: `protocol_id`, `step` (`>= 0`), `instrument_id`, `start_utc` (timezone-aware ISO-8601), `duration_s` (`> 0`), `detector`, `score`, `threshold`.
- Privacy & provenance fields: `sample_rate_hz` (`16000`), `source_sample_rate_hz`, `mic`, `features_path` (`None` or `.npy` log-mel), `speech_band_dropped` (`True`), `raw_audio_kept` (`False`), `raw_audio_saved` (`False`), and outward-rounded `retained_bands_hz` (`[[0, 255], [3464, 8000]]` at `16 kHz` when `speech_band_dropped=True`).

---

## 3. Model

### 3.1 Protocol Span Extractors (`demo/parse_protocol.py` & `protocol/learned_tagger.py`)

1. **Rule-Based Regex Extractor & Unit Normalizer (`demo/parse_protocol.py`)**:
   - Parses integer, European comma-decimal (`0,500 mL`), thousands-grouped (`12,000 x g`), scientific-notation (`2e-3 L`, `1.5 x 10^6`), and signed thermal-range (`-20 to +4 °C`) scalars (`SCALAR`, `NUM`, `TEMP_NUM`) with left-boundary exponent guards (`_NUM_LEFT_GUARD`).
   - Disambiguates bare uppercase `C`/`K` (`_TEMP_BARE_CK`) via thermal verb/preposition context while blocking `MWCO` and `13C`/`14C` NMR isotope tokens (`_NON_THERMAL_AFTER`).
   - Disambiguates bare `<num> g` (`_SPEED_BARE_G`) between reagent mass (`Amount`, e.g., `Add 5 g NaCl`) and centrifugal acceleration (`Speed`, `val >= 50` with centrifugation verbs or `@`/`at` context), suppressing nested `Amount`/`Concentration` matches inside `Speed` (`12000 x g`).
   - Normalizes extracted spans into `ParsedQuantity` (`uL`, `mg`, `mol`, `U`, `C`, `s`, `rpm`, `xg`, `mM`) and scores spans via 1-to-1 maximum bipartite overlap matching (`count_overlap_matches`).

2. **Zero-Leakage Structured Linear-Chain BIOES Tagger (`protocol/learned_tagger.py`)**:
   - Tokenizes lines with an offset-preserving sub-token grammar (`tokenize_protocol_lines`) separating punctuation, degree symbols, numeric literals, and unit stems (`99.65%` exact gold boundary alignment on `WLP/train`).
   - Extracts purely lexical, case-preserved, word-shape, character prefix/suffix (`2..4`), whitespace-adjacency, numeric-magnitude bucket, token/class bigram & trigram, and `[-5, +4]` local clause bag-of-words features (`extract_sequence_features`).
   - Trains a 17-state `BIOES` linear-chain model (`O` + `{B,I,E,S} x {Amount, Temperature, Time, Speed}`) using **cost-augmented Passive-Aggressive (`PA-II`) max-margin updates** with averaged weights and hard legal-transition Viterbi decoding, serialized without `pickle` (`.json.gz`).

### 3.2 Log-Mel Frontend & Hann-Skirt Speech Excision (`acoustic/features.py`)

- **Dtype-Exact WAV Loading (`load_wav_info`, `load_wav`)**: Deterministically maps `uint8`, `int16` (`/ 32768.0`), `int32` (`/ 2147483648.0`), arbitrary integer widths, and `float32` waveforms to mono `float32`, downsampling `> 16 kHz` inputs via `scipy.signal.resample_poly` while blocking `< 16 kHz` upsampling unless `allow_upsample=True`.
- **Two-Stage `[300, 3400] Hz` Speech Excision (`log_mel`, `speech_band_mask`, `retained_mel_bands_hz`)**:
  1. Applies a pre-STFT full-clip FFT notch zeroing `[300.0, 3400.0] Hz` prior to framing so speech-band energy cannot leak across short `64 ms` frame boundaries.
  2. Computes `1024`-point Hann STFT (`hop=512`) and projects onto `64` HTK mel filters across `[0, 8000] Hz`, retaining only filters whose Hann-skirted support `[lo_hz - 2*df, hi_hz + 2*df]` (`2*df = 31.25 Hz`) lies strictly outside `(300, 3400) Hz` (`28` kept bins covering outward-rounded `[[0, 255], [3464, 8000]] Hz`, achieving `>= 60 dB` speech-band suppression).
- **Sub-Clip Slicing & Short-Clip Padding**: `slice_wav_chunks(wav, window_s, hop_s)` slices waveforms into exact streaming windows; `mel_windows()` slices causal mel sub-windows; `stack_frames(mel, context=5)` edge-pads clips shorter than `5` frames.

### 3.3 Unsupervised Detectors, Window-Matched Streaming Calibration & Interlock

1. **One-Class Acoustic Detectors (`acoustic/detect.py`)**:
   - `GaussianDetector`: Fits `LedoitWolf` covariance on per-bin `[mean, std]` vectors (`clip_vector`) and scores by **unsquared** Mahalanobis distance (`squared=False` default) with `pooling in ("mean", "p95", "max_window")`.
   - `KNNDetector(k=5)`: Standardizes vectors via `_Scaler` and computes mean Euclidean distance to `min(k, n_train)` nearest neighbors (with `UserWarning` clamping on few-shot sets).
   - `AutoencoderDetector`: Trains a `128-128-8-128-128` ReLU `MLPRegressor` on up to `60,000` 5-frame stacked windows (`seed=0`) and scores reconstruction MSE with `mean`, `p95`, or sliding `max_window` aggregation.
2. **Window-Matched Out-of-Fold Streaming Calibration (`calibrate_streaming_with_mode`, `acoustic/monitor.py`)**:
   - Holds out entire normal clips in 20-fold (5-fold for autoencoder) cross-validation across **all** normal training clips, slices held-out clips into `(window_s, hop_s)` chunks, and computes conformal order-statistic thresholds (`oof_chunk_conformal`).
   - Stores `calibration_window_s` and `calibration_hop_s` in non-pickle `.npz`/`.json` bundles (`lab-ears/detector-bundle/v1`); `monitor_stream()` raises `ValueError` on `window_s` mismatch unless `allow_window_mismatch=True`.
3. **Cross-Modal Pipetting Interlock (`community_extensions/cross_modal_interlock.py`)**:
   - Monitors voice-stripped `[60, 260] Hz` stepper-stall and `[3600, 7800] Hz` cavitation band ratios inside active `PipettingCommandWindow` transfers.
   - Supports `speech_rejection_mode="bandstop_only"` and `"harmonic_onset_aware"` (which estimates high-band turbulent floors from interharmonic comb valleys `(k + 0.5) * F0` when multi-tooth voiced glottal combs are detected, notches voiced `F0`/`2*F0` in `[60, 260] Hz`, and uses sustained 45th/50th-percentile stroke statistics against 75th-percentile out-of-window background).

---

## 4. Training & Benchmark Results

### 4.1 WLP Protocol Span Extraction (`results/wlp_learned_vs_rules.json` & `demo/output.txt`)

- **Rule-based baseline (`demo/parse_protocol.py`)**: Zero learned parameters, but regular-expression patterns were tuned with `WLP/test` errors visible (`#3`–`#8`, `#28`), explaining why `test` (`0.8502` micro exact F1) scores higher than `train` (`0.8068` micro exact F1; `Amount` `0.7713`).
- **Learned sequence tagger (`protocol/learned_tagger.py`)**: Fit strictly on `WLP/train` (`374` protocols), tuned strictly on `WLP/dev` (`c_pa=0.25, fn_cost=1.30, epochs=14`), and evaluated once on `WLP/test` (`125` protocols):

| Split (`protocols`) | Extractor | `Amount` `F1_ex` / `F1_ov` | `Temperature` `F1_ex` / `F1_ov` | `Time` `F1_ex` / `F1_ov` | `Speed` `F1_ex` / `F1_ov` | Micro `F1_ex` / `F1_ov` |
|---|---|---:|---:|---:|---:|---:|
| `train` (`374`) | Rule-based regex | 0.7713 / 0.8006 | 0.7861 / 0.8315 | 0.8635 / 0.9044 | 0.8429 / 0.8944 | 0.8068 / 0.8447 |
| `train` (`374`) | `LinearChainSpanTagger` | 0.9391 / 0.9493 | 0.9298 / 0.9437 | 0.9584 / 0.9687 | 0.9689 / 0.9807 | **0.9456 / 0.9567** |
| `dev` (`123`) | Rule-based regex | 0.8025 / 0.8406 | 0.8516 / 0.8807 | 0.9042 / 0.9334 | 0.8178 / 0.8972 | 0.8427 / 0.8800 |
| `dev` (`123`) | `LinearChainSpanTagger` | 0.8311 / 0.8624 | 0.9183 / 0.9376 | 0.9257 / 0.9522 | 0.8478 / 0.9087 | **0.8779 / 0.9081** |
| `test` (`125`) | Rule-based regex (`demo/output.txt`) | 0.8022 / 0.8182 | 0.8608 / 0.9084 | 0.9089 / 0.9374 | 0.8372 / 0.9037 | 0.8502 / 0.8798 |
| `test` (`125`) | `LinearChainSpanTagger` (zero-leakage) | **0.8199 / 0.8316** | **0.9134 / 0.9390** | **0.9164 / 0.9463** | **0.9159 / 0.9408** | **0.8761 / 0.8974** |
| `pooled` (`622`) | Rule-based regex | 0.7839 / 0.8121 | 0.8143 / 0.8569 | 0.8812 / 0.9172 | 0.8362 / 0.8966 | 0.8229 / 0.8589 |
| `pooled` (`622`) | `LinearChainSpanTagger` | **0.8924 / 0.9071** | **0.9241 / 0.9415** | **0.9429 / 0.9606** | **0.9334 / 0.9582** | **0.9177 / 0.9347** |

### 4.2 Unsupervised Acoustic Fitting & Streaming Benchmark (`results/dcase2023_cpu.json`)

- **Offline `10 s` DCASE 2023 Task 2 dev (1 CPU thread, `full_hann_skirt_plus_pre_stft_notch`)**:
  - **Fullband (`drop_speech_band=False`)**: Harmonic mean across `{fan, valve, bearing} x {AUC_source, AUC_target, pAUC}` = **`0.5617`** (`autoencoder`), **`0.5479`** (`gaussian`), **`0.5025`** (`knn`); median clip latency `2.4–4.6 ms`.
  - **Speech-band dropped (`drop_speech_band=True`)**: Harmonic mean = **`0.5449`** (`gaussian`), **`0.5169`** (`autoencoder`), **`0.4947`** (`knn`); `valve` `gaussian` reaches `0.7368` source AUC / `0.6896` target AUC / `0.5126` pAUC because impulsive valve clicks concentrate above `3.46 kHz`.
- **Causal `2.0 s` Streaming Evaluation (`valve`, `drop_speech_band=True`, `hop_s=1.0`, `quantile=0.99`, `target_chunk_fpr=0.01`)**:
  - `gaussian`: `normal_chunk_fpr = 0.0133` (`12 / 900` chunks = `1.33x` target), `false_alarms_per_hour = 43.2`, `time_to_first_flag_s_median = 8.5 s`, `chunk_latency_ms_median = 0.59 ms`, `streaming_pauc = 0.5084`.
  - `knn`: `normal_chunk_fpr = 0.0078` (`7 / 900` chunks = `0.78x` target), `false_alarms_per_hour = 25.2`, `time_to_first_flag_s_median = 4.0 s`, `chunk_latency_ms_median = 0.78 ms`, `streaming_pauc = 0.5084`.

---

## 5. History

| Date | Commit / Ref | Summary of Changes |
|---|---|---|
| `2026-11-18` | `ba743be` (`main`) | Initial WLP regex parser (`demo/parse_protocol.py`), `demo/output.txt`, CI workflow, and curated dataset/paper tables. |
| `2026-11-18` | `5476541` (`acoustic-anomaly`) | Added `acoustic/` log-mel + one-class anomaly baseline (`gaussian`, `knn`, `autoencoder`), `AudioEvent` schema, and DCASE 2023 runner. |
| `2026-11-24` | `df7acb6` (`main`) | Canonical unit normalization (`parse_quantities`), discontinuous BRAT handling, 1-to-1 bipartite overlap scoring, dtype-exact WAV loading, and live `acoustic.monitor`. |
| `2026-11-25` | `d3ba222` (`main`) | Added `autolab_wire.py`, Lean 4.36.0 proofs (`lean/LabEarsLean.lean`), Hann-skirt + pre-STFT `[300, 3400] Hz` notch, OOF conformal calibration, non-pickle bundles, and `OpenSensoryDataset` registry. |
| `2026-11-27` | `feat/frontier-streaming-speech-wlp-and-privacy` | Resolved `#36`–`#39` and added frontier upgrades: window-matched `oof_chunk_conformal` 2 s streaming calibration on DCASE `valve`, broadband pulse-train + fricative interlock benchmark & `harmonic_onset_aware` comb-valley rejection, zero-leakage `LinearChainSpanTagger` (`protocol/learned_tagger.py`), fail-closed `autolab_wire.py` privacy/default-WARN gate, and Lean 4.36.0 `#print axioms` CI. |

---

## 6. Unique Properties & Honest Limitations

### Distinctive Invariants & Guarantees

1. **Zero-Bloat Dependency Footprint**: `demo/parse_protocol.py` and `protocol/learned_tagger.py` run on the Python standard library alone (`numpy` optional accelerator); `acoustic/` requires only `numpy`, `scipy`, `scikit-learn`, and `threadpoolctl`.
2. **Fail-Closed Speech-Privacy Architecture**: Pre-STFT `[300, 3400] Hz` spectral notch + outward-rounded Hann-skirt mel exclusion (`[[0, 255], [3464, 8000]] Hz`), `AudioEvent.validate()` invariant checks, Lean 4.36.0 theorems (`LabEarsLean.lean`, verified in CI with `#print axioms`), and fail-closed `autolab_wire.py` rejection of acoustic records missing explicit privacy metadata.
3. **Real-Time Single-Core CPU Execution**: `5.9–6.9 ms` median latency per `10 s` clip (`0.59–0.78 ms` per `2.0 s` streaming chunk) on one CPU thread, with safe `allow_pickle=False` `.npz` and `.json.gz` persistence.

### Known Limitations

- **Regex split leakage vs. entity scope**: The rule-based patterns in `demo/parse_protocol.py` were refined with `WLP/test` errors visible (`Amount` `0.8022` on `test` vs `0.7713` on `train`). `protocol/learned_tagger.py` enforces strict `train -> dev -> test` separation for `Amount`, `Temperature`, `Time`, and `Speed`, while full `Action`/`Reagent` entity recognition and argument-link graph parsing remain future work.
- **Acoustic domain shift & onset labels**: Under DCASE 2023 Task 2 target-domain shift (`10` target training clips), lightweight mel detectors reach `0.4947–0.5617` harmonic mean; pretrained audio embeddings and physical wet-lab instrument recordings are needed for higher target-domain recall. DCASE clips also lack ground-truth anomaly onset timestamps (`time_to_first_flag_s_median` measures clip-relative first alert, not true onset delay).
- **Airborne speech bleed into sub-300 Hz and > 3.6 kHz bands**: Removing `[300, 3400] Hz` (`bandstop_only`) eliminates pure in-band tones (`1.0` F1) but still halts on `67.3%` of clips at `0 dB` clip-RMS under voiced `100–220 Hz` harmonics and `4–8 kHz` fricatives. `harmonic_onset_aware` reduces `0 dB` clip-RMS false halts to `0.0%` (`0.9787` F1), yet loud room-microphone conversation (`+7.6` to `+17.6 dB` above a quiet microstepper) still causes `38.5%–48.1%` false halts—motivating contact microphones on physical liquid handlers. `RHEOLOGY_PRIORS` in `cross_modal_interlock.py` remain heuristic synthetic priors pending physical calibration.

---

## 7. 15-Repository Auto-Lab Wire Contract (`autolab-wire/v1`)

`autolab_wire.py` (`REPO_ID = "lab-ears"`, `LAYER = "L2"`) implements `process_packet(packet: dict) -> dict`:

- **Default Mel Bands**: `DEFAULT_RETAINED_BANDS_HZ` is derived directly from `acoustic.features.retained_mel_bands_hz(drop_speech_band=True)` (`[[0, 255], [3464, 8000]]` at `16 kHz`).
- **Fail-Closed Attached Record Privacy Gate**: Every attached `AudioEvent` (`audio_event`, `audio_events`) or `InterlockDecision` (`interlock_decision`, `interlock_decisions`) must explicitly provide `speech_band_dropped`, `raw_audio_saved` (or `raw_audio_kept`), and `retained_bands_hz`. Missing or violating fields immediately set `halted = True`, `halt_reason = "L2_SPEECH_PRIVACY_BLOCK"`, and name the missing field(s) in `trace[-1]["summary"]`.
- **Explicit Default-Fallback Warning**: When `defaults_applied` is non-empty (`volume_ul=100.0`, `temp_c=25.0`, or `duration_min=15.0` not found in `goal_text`), `trace[-1]["status"]` is set to `"WARN"` (never silent `"PASS"`), and `"defaults_only": True` is emitted when zero quantitative spans were parsed.
- **Fault Propagation**: Active cross-modal cavitation/stall decisions halt with `"L2_CAVITATION_INTERLOCK_HALT"`, and calibrated acoustic threshold exceedances halt with `"L2_ACOUSTIC_ANOMALY_HALT"`.
