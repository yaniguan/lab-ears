# `lab-ears` — Architecture and History

> **Auto-Lab Ecosystem**: [`auto-lab`](https://github.com/yaniguan/auto-lab) · [Project Studio](https://yaniguan.github.io/auto-lab/)  
> **Layer**: `L2 · Auditory & Protocol Sensory Cortex`  
> **License**: MIT (repository code; external datasets retain their respective licenses)

---

## 1. Object

`lab-ears` is the **Auditory & Protocol Sensory Cortex (`L2`)** of the [Robot AI Auto-Lab](https://github.com/yaniguan/auto-lab). Within the `L0`–`L6` Brain/Body architecture, `L2` is responsible for sensory ingestion across two complementary channels before any downstream reasoning, verification, or robotic actuation takes place:

1. **Written & Spoken Protocol Comprehension (`main` — `demo/parse_protocol.py`)**: Extracting structured, checkable experimental parameters (`Amount`, `Temperature`, `Time`, and `Speed` character spans) from free-form wet-lab protocol prose so downstream planners and compilers do not rely on ungrounded free-text numbers.
2. **Unsupervised Machine Acoustic Monitoring (`origin/acoustic-anomaly` — `acoustic/`)**: Listening to benchtop instruments (centrifuges, pumps, fans, valves, bearings) during protocol execution, learning a one-class representation of normal operating sound per machine without anomalous training examples, and emitting step-indexed anomaly records (`lab-ears/audio-event/v1`) while stripping human speech frequencies (`300–3400 Hz`) and discarding raw waveforms by default.

### Inputs and Outputs

| Track | Branch | Input | Output |
|---|---|---|---|
| Protocol span extraction | `main` (`ba743be`) | Plain-text wet-lab protocol files (`*.txt`) and optional BRAT standoff gold files (`*.ann`) | Set of `(label, start_char, end_char)` spans (`find_spans(text)`) and corpus-level `P_exact`, `R_exact`, `F1_exact`, `P_overlap` table |
| Machine acoustic anomaly detection | `origin/acoustic-anomaly` (`5476541`) | `16 kHz` WAV audio clips (`scipy.io.wavfile`) and step metadata (`protocol_id`, `step`, `instrument_id`) | `(n_frames, n_mels)` float32 log-mel spectrograms, scalar anomaly scores, DCASE 2023 `AUC_source`/`AUC_target`/`pAUC` JSON reports, and `lab-ears/audio-event/v1` JSONL ledger lines |

### Scope Boundaries

- **Implemented today**: Zero-dependency regular-expression baseline for four quantitative entity types on the Wet Lab Protocols (WLP) corpus (`main`), plus a `numpy`/`scipy`/`scikit-learn` unsupervised acoustic anomaly pipeline (`gaussian`, `knn`, `autoencoder`) with speech-band masking and run-ledger event serialization (`origin/acoustic-anomaly`).
- **Out of scope / next stages**: Learned sequence labeling for `Action` and `Reagent` entities and their argument links; automatic speech recognition (ASR via Whisper / `faster-whisper` / `sherpa-onnx`) for spoken benchtop goals; wet-lab-specific normal audio corpora recorded directly on real centrifuges, liquid handlers, and peristaltic pumps.

---

## 2. Data

### 2.1 Active Benchmark Datasets

1. **Wet Lab Protocols Corpus (WLP)** — [`chaitanya2334/WLP-Dataset`](https://github.com/chaitanya2334/WLP-Dataset) (Kulkarni et al., NAACL 2018, `arXiv:1805.00195`, **MIT License**):
   - `622` biology and chemistry wet-lab protocols annotated in BRAT standoff format (`*.txt` raw protocol text paired with `*.ann` entity/relation annotations).
   - `demo/parse_protocol.py` evaluates on the `125`-protocol `test` split (`data/WLP-Dataset/test`), scoring contiguous `T*` entity lines (`";" not in info`) across four target labels:
     - `Amount`: `1,011` gold spans
     - `Time`: `810` gold spans
     - `Temperature`: `509` gold spans
     - `Speed`: `155` gold spans
   - Not stored in git (`data/` is gitignored); fetched on demand via shallow clone in local runs and CI.

2. **DCASE 2023 Challenge Task 2 Development Set** — [Zenodo Record 7882613](https://zenodo.org/records/7882613) (**CC BY 4.0**, `origin/acoustic-anomaly`):
   - First-shot unsupervised anomalous sound detection under domain shift across 7 machine types (`bearing`, `fan`, `gearbox`, `slider`, `ToyCar`, `ToyTrain`, `valve`).
   - Default evaluation (`acoustic/fetch_dcase2023.py` and `acoustic/evaluate.py`) downloads `fan`, `valve`, and `bearing` (~`270–390 MB` zipped per machine, ~`850 MB` total; extracted under gitignored `data/dcase2023/`).
   - Per machine: `1,000` normal `10 s` `16 kHz` WAV clips in `train/` (`990` source-domain clips + `10` target-domain clips; zero anomalies) and `200` `10 s` test clips in `test/` with ground-truth domain (`_source_` vs target) and condition (`_anomaly_` vs normal) encoded in filenames.

3. **Synthetic Harmonic + Knock Unit-Test Corpus** (`tests/test_acoustic.py` on `origin/acoustic-anomaly`):
   - Generated deterministically in memory (`np.random.default_rng(0)`): `2.0 s` clips at `16 kHz` with jittered fundamental $f_0 \approx 120\text{ Hz}$, harmonics $(1, 0.5), (2, 0.2), (3, 0.1)$, additive Gaussian noise ($\sigma = 0.02$), and anomalous broadband exponentially decaying clicks every `0.25 s` (`knock=True`). `30` normal training clips + `20` test clips (`10` normal, `10` knock).

### 2.2 Event Schema (`lab-ears/audio-event/v1`)

Defined in `acoustic/record.py` (`AudioEvent` dataclass) on `origin/acoustic-anomaly` to join acoustic monitoring with run ledgers:

- Required fields: `protocol_id: str`, `step: int` ($\ge 0$), `instrument_id: str`, `start_utc: str` (ISO-8601 UTC string), `duration_s: float`, `detector: str`, `score: float`, `threshold: float`.
- Defaults & privacy metadata: `sample_rate_hz: int = 16000`, `mic: str = "unknown"` (e.g., `"contact"`, `"near-field"`, `"room"`), `features_path: Optional[str] = None`, `speech_band_dropped: bool = True`, `raw_audio_kept: bool = False`, `notes: str = ""`, `schema: str = "lab-ears/audio-event/v1"`.
- Derived field serialized in `to_json()`: `"flagged": score > threshold` (stripped on `from_json()` round-trip).

### 2.3 Curated Reference Datasets

`README.md` catalogs external datasets for protocol understanding, machine acoustics, and ASR with explicit commercial-use warnings:
- **Permissive / Share-Alike**: WLP (`MIT`), LAB-Bench (`CC BY-SA 4.0`), DCASE 2023 Task 2 dev (`CC BY 4.0`), MIMII (`CC BY-SA 4.0`), LibriSpeech (`CC BY 4.0`), Common Voice (`CC0`), `protocols.io` (per-protocol, typically `CC BY`).
- **Restricted / Unlicensed**: BioProBench (`CC BY-NC 4.0`, **non-commercial** research only), BioProt / BioPlanner (no license file; verify before use).

---

## 3. Model

### 3.1 Rule-Based Protocol Span Extractor (`demo/parse_protocol.py`)

A deterministic regular-expression span recognizer built solely on Python's `re` module:

- **Shared numeric literal & range sub-pattern**:
  ```python
  NUM = r"(?:\d+(?:[.,]\d+)?(?:\s*(?:-|–|to)\s*\d+(?:[.,]\d+)?)?)"
  ```
  Matches integers, dot- or comma-decimals, and hyphen/en-dash/`to` ranges (e.g., `10-20`, `2.5 to 5`).
- **Slot grammars (`RULES`)**:
  - `Amount`: `NUM` followed by volume (`µl`, `μl`, `ul`, `ml`, `mL`, `µL`, `μL`, `uL`, `L`, `l`, `nl`, `nL`, `volumes?`, `vol`), mass (`mg`, `µg`, `μg`, `ug`, `ng`, `g`, `kg`), or molar (`pmol`, `nmol`, `µmol`, `mmol`, `mol`) units with word boundary `\b`.
  - `Temperature`: `NUM` followed by `°C` / `ºC` / `˚C` / `degrees Celsius` / `C` / `°F` / `K`, or qualitative thermal states `\b(?:room temperature|RT|on ice|ice)\b`.
  - `Time`: `NUM` followed by `h` / `hr` / `hrs` / `hours?` / `min` / `mins` / `minutes?` / `s` / `sec` / `secs` / `seconds?` / `days?` / `overnight`, or standalone `\bovernight\b`.
  - `Speed`: `NUM` followed by `x g` / `× g` / `g\b` / `rpm` / `RPM` / `xg`.
- **Case sensitivity and span disambiguation (`find_spans`)**:
  - `Amount`, `Temperature`, and `Time` run with `re.IGNORECASE`; `Speed` runs case-sensitively (`flags=0`) so uppercase `G` (e.g., gauge) does not trigger centrifugal `g`.
  - Because strings such as `"1000g"` match both `Speed` (`g\b`) and `Amount` (`g\b`) with identical `(start, end)` offsets, `find_spans` drops `(Amount, s, e)` whenever `(s, e)` is also present in `Speed`.

### 3.2 Log-Mel & Privacy Filter (`acoustic/features.py` on `origin/acoustic-anomaly`)

Implemented in pure `numpy` and `scipy.io.wavfile` without `librosa` or `torchaudio`:

- **Audio loading (`load_wav`)**: Enforces `SR = 16000` Hz, converts multi-channel audio to mono via `x.mean(axis=1)`, and scales PCM16 waveforms (`max(|x|) > 1.5`) by `1 / 32768.0` into `float32`.
- **STFT & Mel Filterbank (`mel_filterbank`, `log_mel`)**:
  - `N_FFT = 1024` (`64 ms` Hanning window `_WINDOW = np.hanning(1024)`), `HOP = 512` (`32 ms` hop, `311` frames per `10 s` clip), `N_MELS = 64` triangular filters spaced uniformly on the HTK mel scale $m(f) = 2595 \log_{10}(1 + f / 700)$ across `[0, 8000]` Hz.
  - Power spectrum $S = |\text{rfft}(x_{\text{framed}} \odot w)|^2$ projected to decibels: $M = 10 \log_{10}(S F^\top + 10^{-10}) \in \mathbb{R}^{T \times 64}$.
- **Speech-band excision (`speech_band_mask`)**:
  - Drops mel bins whose center frequencies fall inside `SPEECH_BAND_HZ = (300.0, 3400.0)` via `(centers_hz < 300.0) | (centers_hz > 3400.0)`, retaining low-frequency mechanical hum (`< 300 Hz`) and high-frequency bearing/valve hiss (`> 3400 Hz`).
- **Feature aggregation**:
  - `clip_vector(mel)`: Concatenates temporal mean and standard deviation per mel bin ($\mathbb{R}^{2D}$, `128` dims fullband).
  - `stack_frames(mel, context=5)`: Concatenates `5` consecutive mel frames into sliding windows of shape `(T - 4, 5 * D)` (`320` dims fullband, matching the DCASE baseline input structure).

### 3.3 Unsupervised Anomaly Detectors (`acoustic/detect.py` on `origin/acoustic-anomaly`)

All three detectors implement `.fit(mels)` on normal clips only and `.score(mel) -> float` (higher = more anomalous):

1. **`GaussianDetector`**: Fits shrinkage-regularized empirical covariance `sklearn.covariance.LedoitWolf` on `clip_vector(mel)` across normal clips; scores test clips by squared Mahalanobis distance `cov.mahalanobis(...)`.
2. **`KNNDetector(k=5)`**: Standardizes `clip_vector(mel)` via `_Scaler` ($\mu, \sigma + 10^{-6}$), fits `NearestNeighbors(n_neighbors=5)` with Euclidean distance, and returns the mean distance to the `5` nearest normal training clips.
3. **`AutoencoderDetector`**: Stacks `context=5` frames across all training clips, subsamples up to `max_frames=60000` windows without replacement (`default_rng(seed=0)`), standardizes via `_Scaler`, and trains `MLPRegressor(hidden_layer_sizes=(128, 128, 8, 128, 128), max_iter=30, batch_size=512, learning_rate_init=1e-3, random_state=0, early_stopping=False)` to reconstruct its input `Xs -> Xs`. Anomaly score is the clip-mean reconstruction MSE $\frac{1}{N D}\sum \|\hat{X}_s - X_s\|_2^2$.

---

## 4. Training

### 4.1 Protocol Span Extraction (`main` — Rule-Based, No Learned Weights)

`demo/parse_protocol.py` involves **no parameter training**; regex rules were authored against standard wet-lab unit conventions and evaluated on the held-out `125`-protocol WLP `test` split (`demo/output.txt`, measured on CPU, 2026-11-18):

| Label | Gold Spans | Predicted Spans | `P_exact` | `R_exact` | `F1_exact` | `P_overlap` |
|---|---:|---:|---:|---:|---:|---:|
| `Amount` | 1,011 | 1,103 | 0.756 | 0.825 | 0.789 | 0.773 |
| `Temperature` | 509 | 584 | 0.724 | 0.831 | 0.774 | 0.839 |
| `Time` | 810 | 806 | 0.909 | 0.905 | 0.907 | 0.940 |
| `Speed` | 155 | 182 | 0.692 | 0.813 | 0.748 | 0.747 |

`Time` reaches `0.907` exact F1 (`0.940` overlap precision) with plain regular expressions, whereas `Speed` (`0.748`) and `Temperature` (`0.774`) suffer false positives from bare unit tokens (`C`, `K`, `g`, `ice`) and unmodeled context—motivating a learned sequence tagger for full action-reagent-parameter graph extraction.

### 4.2 Unsupervised Acoustic Fitting (`origin/acoustic-anomaly`)

- **Strict one-class protocol**: Detectors fit exclusively on `1,000` normal clips per machine (`990` source + `10` target domain) on a single CPU thread (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`).
- **Measured DCASE 2023 Task 2 Dev Results (`results/dcase2023_cpu.json`, Intel Xeon `x86_64`, 1 thread, 2026-11-18)**:
  - **Full spectrum (`drop_speech_band=False`)**:
    - `autoencoder`: harmonic mean across `{fan, valve, bearing} × {AUC_source, AUC_target, pAUC@0.1}` = **`0.5548`** (`AUC_source`: `0.7188` fan, `0.5760` valve, `0.7064` bearing; `AUC_target`: `0.4976`, `0.4492`, `0.5140`; fit `18.3–19.8 s`; median clip latency `3.3–5.7 ms`).
    - `gaussian`: harmonic mean = **`0.5482`** (`AUC_source`: `0.4748` fan, `0.6856` valve, `0.6888` bearing; fit `0.1 s`; median clip latency `2.8–5.3 ms`).
    - `knn`: harmonic mean = **`0.5023`** (fit `0.1 s`; median clip latency `2.9–4.0 ms`).
  - **Speech band dropped (`drop_speech_band=True`, `300–3400 Hz` removed)**:
    - Harmonic means shift to **`0.5228`** (`gaussian`), **`0.4962`** (`autoencoder`), and **`0.4868`** (`knn`). Removing `300–3400 Hz` degrades `fan` `autoencoder` `AUC_source` from `0.7188` to `0.3204` (blade-pass harmonics fall inside the speech band) while improving `valve` `gaussian` `AUC_source` from `0.6856` to `0.7320` (`AUC_target` `0.5268 -> 0.6696`) because impulsive valve clicks concentrate above `3.4 kHz`.
- **Honest scientific assessment**: Under DCASE 2023 domain shift (`10` target training clips), target-domain AUC across all three lightweight baselines hovers near chance (`0.45–0.54` fullband), consistent with the official DCASE 2023 GPU autoencoder baseline (`0.362` target AUC on fan, `0.507` on valve, `0.558` on bearing). Inference speed is already production-ready (`< 6 ms` median per `10 s` clip on 1 CPU core); representation quality under domain shift (e.g., pretrained audio embeddings + kNN) and real lab-instrument recordings are the true bottlenecks.

---

## 5. History

| Date (ISO-8601) | Commit | Branch / Ref | Summary of Changes |
|---|---|---|---|
| `2026-11-18 01:35:42 -0800` | `ba743be` (`ba743beff43aeebac7f08c8183fccfe75f5b075f`) | `main` | **Initial baseline (`Add curated resources and first demo`)**: Added zero-dependency WLP regex parser (`demo/parse_protocol.py`), recorded 125-protocol WLP test evaluation (`demo/output.txt`), added GitHub Actions CI (`.github/workflows/demo.yml`), and curated protocol/ASR datasets, open-source repositories, and papers in `README.md`. |
| `2026-11-18 02:04:02 -0800` | `5476541` (`5476541d0c6f128e46aa61b9df2495a9b2a5df73`) | `origin/acoustic-anomaly` (open PR branch off `ba743be`) | **Acoustic anomaly baseline (`Add machine sound anomaly detection baseline`)**: Added `acoustic/` package (`features.py`, `detect.py`, `evaluate.py`, `fetch_dcase2023.py`, `record.py`, `requirements.txt`), single-thread CPU benchmark results on DCASE 2023 Task 2 (`results/dcase2023_cpu.json`), synthetic knock unit tests (`tests/test_acoustic.py`), and the `acoustic` CI job in `.github/workflows/demo.yml`. |

---

## 6. Unique Properties

### Distinctive Invariants & Guarantees

1. **Zero-Bloat Dependency Footprint**: `main` uses the Python standard library exclusively (`requirements.txt` contains `# standard library only`); `origin/acoustic-anomaly` requires only `numpy`, `scipy`, and `scikit-learn`, avoiding FFmpeg/libsndfile/PyTorch system-library fragility on lab benchtop controllers.
2. **Privacy-by-Construction Ledger Contract**: `AudioEvent` defaults to `raw_audio_kept=False` and `speech_band_dropped=True`, and `AudioEvent.validate()` enforces a hard invariant raising `ValueError("raw audio still contains the speech band")` if a caller attempts to mark `speech_band_dropped=True` while retaining raw audio (`raw_audio_kept=True`).
3. **Sub-20 ms Single-Core CPU Latency**: End-to-end WAV file read, 1024-point STFT, 64-bin mel projection, and detector scoring take `2.8–5.7 ms` median (`3.0–17.9 ms` p95) per `10 s` audio clip on a single CPU thread—over `1,700x` faster than real time.

### Known Limitations

- `gold_spans()` in `demo/parse_protocol.py` ignores discontinuous BRAT annotations (`";" not in info`) and reports `P_overlap` without `R_overlap` / `F1_overlap`.
- Single-letter units (`C`, `K`, `L`, `l`, `g`) and uncontextualized words (`ice`, `RT`) cause false-positive spans in reagent names and catalog identifiers (e.g., `"Protein G"`, `"100 K"`, `"RNase H"`).
- On `origin/acoustic-anomaly`, `load_wav()` only normalizes integer PCM16 (`max > 1.5 -> / 32768.0`), mis-scaling 24-bit/32-bit integer WAVs (`int32`) or quiet PCM16 clips (`|x| <= 1`), and `stack_frames()` raises `ValueError` if a clip has fewer than `context=5` frames (`< 160 ms`).

### Ecosystem Interfaces Across Auto-Lab

- **Downstream to `L1` (`stepbot`, `twinstep`)**: Extracted `Amount`, `Temperature`, `Time`, and `Speed` spans ground free-text wet-lab procedures into typed liquid-handling YAML/JSON steps for [`stepbot`](https://github.com/yaniguan/stepbot) and protocol verification benchmarks in [`twinstep`](https://github.com/yaniguan/twinstep).
- **Downstream to `L3` (`YLM`) & `L4` (`lab-rehearsal`, `lab-verify`, `lab-attrib`)**: Structured protocol slots feed [`YLM`](https://github.com/yaniguan/YLM) multimodal reasoning, while `lab-ears/audio-event/v1` JSONL events join [`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal)'s `runs.jsonl` execution ledger, [`lab-verify`](https://github.com/yaniguan/lab-verify)'s safety gates, and [`lab-attrib`](https://github.com/yaniguan/lab-attrib)'s post-execution causal blame attribution to pause or quarantine runs when an instrument sounds abnormal.
- **Downstream to `L0` (`lab-feet`)**: Acoustic autoencoders and future small Whisper/audio-embedding models target [`lab-feet`](https://github.com/yaniguan/lab-feet) for `int8`/`4-bit` edge deployment on contact-microphone boards mounted directly on lab instruments.

---

## 7. 15-Repository Auto-Lab Wire Contract (`autolab-wire/v1`)

`autolab_wire.py` (`REPO_ID = "lab-ears"`, `LAYER = "L2"`) implements `process_packet(packet: dict) -> dict` as the `L2` sensory and protocol-comprehension entry point of the 15-repository Auto-Lab wire bus:

- **Path & State Discipline**: Resolves all repository-local resources strictly via `REPO_DIR = Path(__file__).resolve().parent` (never `os.getcwd()`), deep-copies incoming packets, initializes `autolab-wire/v1` envelope defaults (`schema_version`, `halted`, `halt_reason`, `trace`, `bottom_reasoning_traces`), and short-circuits unchanged when `halted` is already `True`.
- **Step-Indexed Protocol Span Normalization & Range Bounds**: Invokes `demo/parse_protocol.py` (`parse_quantities(..., include_concentration=True)`) on `goal_text` (handling European comma decimals, scientific notation, signed thermal ranges, and `13C NMR` isotope disambiguation), segments ordered protocol clauses into `parsed_spans["steps"]`, anchors top-level `volume_ul` (`uL`, default `100.0`), `temp_c` (`C`, default `25.0`), and `duration_min` (`min`, default `15.0`) to the primary reaction/incubation step while preserving `canonical_min`/`canonical_max` intervals and recording fallback defaults in `defaults_applied`.
- **Lean Speech-Band, Nyquist & Hann-Skirt Gate (`full_skirt_speech_band_drop`)**: Harmonizes `raw_audio_saved` and `raw_audio_kept`, verifies non-empty `retained_bands_hz` (default `[[60, 260], [3600, 7800]]`), checks Nyquist realizability (`2 * hi <= sample_rate_hz`) and Hann-skirt separation (`hi + window_skirt_hz <= 300` or `lo - window_skirt_hz >= 3400`, matching `LabEarsLean.checkLabEars`), and halts privacy violations with `halt_reason = "L2_SPEECH_PRIVACY_BLOCK"`.
- **Calibrated Acoustic Anomaly & Cross-Modal Interlock Propagation**: Inspects attached `AudioEvent` (`lab-ears/audio-event/v1`) and `InterlockDecision` records, emitting normalized `out["acoustic_telemetry"]` and halting active faults with `"L2_CAVITATION_INTERLOCK_HALT"` or `"L2_ACOUSTIC_ANOMALY_HALT"` (live benchtop interlock and streaming ring-buffer monitoring verified on CPU synthetic test suites; offline DCASE 2023 Task 2 baseline measured on single-thread CPU).
- **Upstream / Downstream Links (`lab-ears -> lab-ask-why, stepbot`)**:
  - **Upstream**: Scientist/operator `goal_text`, benchtop acoustic filter configuration, and calibrated `AudioEvent` / `InterlockDecision` telemetry.
  - **Downstream**: Passes `parsed_spans`, `acoustic_telemetry`, and `speech_band_dropped` to `lab-ask-why` (`L6` epistemic contradiction probing), `stepbot`/`lab-hands` (`L1` liquid-handler compilation & mid-step halt), `labshift` (`L5` incubation window scheduling), `lab-rehearsal`/`lab-verify`/`lab-attrib` (`L4` twin simulation, formal gate & causal blame), and `twinstep` (`L3` protocol dependency DAGs).
- **Full 15-Repository Stack**:
  1. `L2` Sensory & Protocol Cortex: `lab-ears`
  2. `L6` Epistemic & Conformal Brain: `lab-ask-why`, `lab-decide`
  3. `L5` Ontology, Make-vs-Buy & Scheduling: `lab-automation-atlas`, `buy-or-make`, `labshift`
  4. `L4` Pre-Run Twin, Formal Gate & Post-Run Attribution: `lab-rehearsal`, `lab-verify`, `lab-attrib`
  5. `L1` Protocol Compilation & Robotic Kinematics: `stepbot`, `lab-hands`
  6. `L0` Quantized Edge Execution: `lab-feet`
  7. `L3` Error Twins, Sequence Packing & RLVR Post-Training: `twinstep`, `labfill`, `labbench-posttrain`

