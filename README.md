# lab-ears

Teach a lab agent to *listen*: turn spoken goals, written protocols and instrument logs into clear, checkable steps, and notice when a machine sounds wrong.

## Where it fits

In the lab agent on [yaniguan.github.io](https://yaniguan.github.io/), the ears are planned. They take chemistry and biology protocols, a scientist's goal by voice or chat, and instrument logs and alarms, and pass clear steps to the brain (YLM, a model that reads text and 3D molecular geometry together). This repo covers two sensory tracks: extracting and normalizing quantitative spans from wet-lab protocol text (both a zero-dependency rule-based baseline and a zero-leakage structured linear-chain BIOES sequence tagger) and unsupervised acoustic anomaly / cross-modal pipetting fault monitoring with speech-band privacy filtering.

## Protocol Span Extraction: Rule-Based Baseline vs. Zero-Leakage Learned BIOES Tagger

### 1. Rule-Based Regex Baseline (`demo/parse_protocol.py`)

A zero-dependency regular-expression extractor (`demo/parse_protocol.py`, standard library only) finds **Amount, Temperature, Time and Speed** (plus optional **Concentration**) spans in wet-lab protocols, normalizes them into canonical Auto-Lab units (`parse_quantities()`), and scores them against the Wet Lab Protocols (WLP) corpus (622 protocols across `train`/`dev`/`test`, commit `1f51ae6e620a5aa3590acf725713f430783f56d7`). Overlap metrics (`P_overlap`, `R_overlap`, `F1_overlap`) use 1-to-1 maximum bipartite matching.

> **Important split-hygiene note**: The regular-expression rules in `demo/parse_protocol.py` were iteratively refined while inspecting `WLP/test` errors (Issues `#3`–`#8`, `#28`), so `WLP/test` is **not** a strictly held-out split for the regex baseline. On the larger `WLP/train` split (`374` protocols), regex `Amount` exact F1 drops to `0.771` (`Temperature` `0.786`). CI diffs `python demo/parse_protocol.py data/WLP-Dataset/test` directly against `demo/output.txt` so committed numbers cannot drift.

```bash
git clone https://github.com/chaitanya2334/WLP-Dataset data/WLP-Dataset
git -C data/WLP-Dataset checkout 1f51ae6e620a5aa3590acf725713f430783f56d7
python demo/parse_protocol.py data/WLP-Dataset/test --min-f1 0.78
```

Exact output (`demo/output.txt`, pinned WLP `test` split, CPU only):

```text
protocols: 125 (discontinuous_gold: 2, skipped_discontinuous: 0)
label           gold  pred  P_exact  R_exact F1_exact  P_overlap  R_overlap F1_overlap
Amount          1013  1121    0.764    0.845    0.802      0.779      0.862      0.818
Temperature      509   583    0.806    0.923    0.861      0.851      0.974      0.908
Time             812   801    0.915    0.903    0.909      0.944      0.931      0.937
Speed            155   146    0.863    0.813    0.837      0.932      0.877      0.904
```

### 2. Frontier Zero-Leakage Linear-Chain BIOES Span Tagger (`protocol/learned_tagger.py`)

To eliminate test-split rule tuning, `protocol/learned_tagger.py` implements `LinearChainSpanTagger`: an offset-preserving sub-token sequence tagger trained with cost-augmented Passive-Aggressive (`PA-II`) max-margin structured updates and constrained 17-state `BIOES` Viterbi decoding over purely lexical, orthographic, numeric-magnitude, and local-window features (`273,555` features, safe non-pickle `.json.gz` serialization, pure-Python fallback when `numpy` is absent).

- **Strict zero-leakage protocol**: Vocabulary and weights are fit **exclusively** on `WLP/train` (`374` protocols); hyperparameters (`c_pa=0.25, fn_cost=1.30, epochs=14`) were selected **exclusively** on `WLP/dev` (`123` protocols); and `WLP/test` (`125` protocols) was evaluated once (`results/wlp_learned_vs_rules.json`).
- **Upstream WLP duplicate audit**: SHA-256 content auditing in `evaluate_folder()` also detects `2` `dev` protocols and `4` `test` protocols in upstream `chaitanya2334/WLP-Dataset` whose raw `.txt` text duplicates a `train` protocol; on the `121` strictly unseen-SHA256 `test` protocols, `LinearChainSpanTagger` still achieves **`0.8751` exact / `0.8965` overlap micro-F1**.

```bash
python -m protocol.learned_tagger \
    --train data/WLP-Dataset/train \
    --dev data/WLP-Dataset/dev \
    --test data/WLP-Dataset/test \
    --tune-on-dev \
    --external-jsonl data/bioprobench/protocols.jsonl data/bioplanner_bioprot/protocols.jsonl \
    --out results/wlp_learned_vs_rules.json
```

### Side-by-Side WLP Results Across All Splits (`results/wlp_learned_vs_rules.json`)

| Split (`protocols`) | Label | Gold | Rule-Based Pred | Rule-Based `F1_exact` / `F1_overlap` | Learned BIOES Pred | Learned BIOES `F1_exact` / `F1_overlap` |
|---|---|---:|---:|---:|---:|---:|
| **`train` (`374`)** | `Amount` | 2,814 | 3,339 | 0.7713 / 0.8006 | 2,885 | **0.9391 / 0.9493** |
| | `Temperature` | 1,400 | 1,775 | 0.7861 / 0.8315 | 1,478 | **0.9298 / 0.9437** |
| | `Time` | 2,100 | 2,199 | 0.8635 / 0.9044 | 2,176 | **0.9584 / 0.9687** |
| | `Speed` | 591 | 536 | 0.8429 / 0.8944 | 598 | **0.9689 / 0.9807** |
| | **Micro (`train`)** | **6,905** | **7,849** | **0.8068 / 0.8447** | **7,137** | **0.9456 / 0.9567** |
| **`dev` (`123`)** | `Amount` | 977 | 1,074 | 0.8025 / 0.8406 | 941 | **0.8311 / 0.8624** |
| | `Temperature` | 460 | 571 | 0.8516 / 0.8807 | 470 | **0.9183 / 0.9376** |
| | `Time` | 679 | 688 | 0.9042 / 0.9334 | 680 | **0.9257 / 0.9522** |
| | `Speed` | 215 | 213 | 0.8178 / 0.8972 | 212 | **0.8478 / 0.9087** |
| | **Micro (`dev`)** | **2,331** | **2,546** | **0.8427 / 0.8800** | **2,303** | **0.8779 / 0.9081** |
| **`test` (`125`)** | `Amount` | 1,013 | 1,121 | 0.8022 / 0.8182 | 1,036 | **0.8199 / 0.8316** |
| | `Temperature` | 509 | 583 | 0.8608 / 0.9084 | 507 | **0.9134 / 0.9390** |
| | `Time` | 812 | 801 | 0.9089 / 0.9374 | 790 | **0.9164 / 0.9463** |
| | `Speed` | 155 | 146 | 0.8372 / 0.9037 | 166 | **0.9159 / 0.9408** |
| | **Micro (`test`)** | **2,489** | **2,651** | **0.8502 / 0.8798** | **2,499** | **0.8761 / 0.8974** |
| **`pooled` (`622`)** | `Amount` | 4,804 | 5,534 | 0.7839 / 0.8121 | 4,862 | **0.8924 / 0.9071** |
| | `Temperature` | 2,369 | 2,929 | 0.8143 / 0.8569 | 2,455 | **0.9241 / 0.9415** |
| | `Time` | 3,591 | 3,688 | 0.8812 / 0.9172 | 3,646 | **0.9429 / 0.9606** |
| | `Speed` | 961 | 895 | 0.8362 / 0.8966 | 976 | **0.9334 / 0.9582** |
| | **Micro (`pooled`)** | **11,725** | **13,046** | **0.8229 / 0.8589** | **11,939** | **0.9177 / 0.9347** |

Even against regex rules that were previously tuned with `WLP/test` errors visible, the zero-leakage `LinearChainSpanTagger` improves held-out `WLP/test` exact micro-F1 from **`0.8502` to `0.8761`** (`+2.59` points), lifting `Speed` (`0.8372 -> 0.9159`, capturing qualitative phrases like `maximum speed` / `full speed`) and `Temperature` (`0.8608 -> 0.9134`, cutting bare-token false positives).

**Zero-Shot External Corpus Transfer (`BioProBench` & `BioPlanner`)**:
- **`BioProBench` (`PQA`, `1,200` protocols)**: Extracts `1,495` normalized spans across `720` quantitative protocols (`607` `Amount`, `356` `Temperature`, `419` `Time`, `113` `Speed`), with `0.8712` exact / `0.8775` overlap cross-extractor agreement against `demo/parse_protocol.py`.
- **`BioPlanner` (`BioProt`, `8` full `protocols.io` SOPs)**: Extracts `107` spans across `8/8` protocols (`46` `Amount`, `20` `Temperature`, `34` `Time`, `7` `Speed`), with `0.8814` exact / `0.8898` overlap agreement.

## Machine Sounds (`acoustic/`)

Instruments often sound wrong before they fail: a worn bearing grinds, an unbalanced rotor knocks, a pump stalls. `acoustic/` learns what normal sounds like for one machine and flags clips or live streaming windows that do not fit. It never sees an anomaly in training.

- **Features**: 64-bin HTK log-mel spectrogram (`16 kHz`, `1024`-point Hann STFT, hop `512`), `numpy` and `scipy` only, dtype-exact PCM scaling (`uint8`/`int16`/`int32`/`float32`).
- **Detectors**: `gaussian` (unsquared Mahalanobis distance with `LedoitWolf` shrinkage covariance), `knn` (mean Euclidean distance to the `5` nearest standardized normal clips), `autoencoder` (dense `128-128-8-128-128` on `5` stacked frames).
- **Data**: `fan`, `valve`, and `bearing` from the [DCASE 2023 Task 2 development set](https://zenodo.org/records/7882613) (`CC-BY-4.0`, `1,000` normal `10 s` training clips + `200` `10 s` test clips per machine). No audio is committed to git.

```bash
pip install -r acoustic/requirements.txt
python -m acoustic.fetch_dcase2023 --machines fan valve bearing
python -m acoustic.evaluate --data data/dcase2023 --machines fan valve bearing
python -m acoustic.evaluate --data data/dcase2023 --machines valve --speech-band drop --detectors gaussian knn --streaming-window-s 2.0
python -m pytest -p no:cacheprovider -v
```

### Offline 10 s Clip Benchmark (`results/dcase2023_cpu.json`, 1 CPU thread)

| Machine | Detector | Fullband `AUC_src` / `AUC_tgt` / `pAUC` | Speech-Dropped `AUC_src` / `AUC_tgt` / `pAUC` | Fit Time (Full / Drop) | Latency per 10 s Clip (Median / p95, Drop) |
|---|---|---:|---:|---:|---:|
| `fan` | `gaussian` | 0.4744 / 0.4984 / 0.5505 | 0.5160 / 0.4476 / 0.5274 | 0.1 / 0.1 s | 6.0 / 6.3 ms |
| `fan` | `knn` | 0.4036 / 0.4612 / 0.5174 | 0.4832 / 0.4012 / 0.5016 | 0.1 / 0.1 s | 6.4 / 6.9 ms |
| `fan` | `autoencoder` | 0.7596 / 0.5184 / 0.5900 | 0.4676 / 0.4076 / 0.5853 | 15.2 / 10.1 s | 6.7 / 7.4 ms |
| `valve` | `gaussian` | 0.6848 / 0.5252 / 0.5011 | 0.7368 / 0.6896 / 0.5126 | 0.1 / 0.1 s | 5.9 / 6.3 ms |
| `valve` | `knn` | 0.5848 / 0.4520 / 0.4911 | 0.6348 / 0.4704 / 0.5000 | 0.1 / 0.1 s | 6.5 / 8.2 ms |
| `valve` | `autoencoder` | 0.5688 / 0.4552 / 0.5200 | 0.5360 / 0.5604 / 0.5095 | 17.0 / 10.6 s | 6.4 / 7.5 ms |
| `bearing` | `gaussian` | 0.6888 / 0.5440 / 0.5421 | 0.5272 / 0.5332 / 0.5211 | 0.1 / 0.1 s | 6.2 / 6.6 ms |
| `bearing` | `knn` | 0.6608 / 0.5128 / 0.5232 | 0.5432 / 0.4628 / 0.5174 | 0.1 / 0.0 s | 6.9 / 8.1 ms |
| `bearing` | `autoencoder` | 0.7184 / 0.5248 / 0.5247 | 0.5584 / 0.5440 / 0.5347 | 16.7 / 11.3 s | 6.7 / 7.2 ms |

Harmonic mean across the nine metrics (`{fan, valve, bearing} x {AUC_source, AUC_target, pAUC}`):
- **Fullband (`drop_speech_band=False`)**: `autoencoder` **`0.5617`**, `gaussian` **`0.5479`**, `knn` **`0.5025`**.
- **Speech-band dropped (`drop_speech_band=True`, `full_hann_skirt_plus_pre_stft_notch`)**: `gaussian` **`0.5449`**, `autoencoder` **`0.5169`**, `knn` **`0.4947`**.

### Window-Matched 2.0 s Causal Streaming Calibration (`valve`, `--speech-band drop`, Issue #36)

Streaming thresholds are calibrated via clip-grouped 20-fold out-of-fold conformal calibration (`oof_chunk_conformal`) across **all `1,000` normal training clips** sliced at the exact same `(window_s=2.0, hop_s=1.0)` window scored at stream time (`9,000` OOF normal chunks). Serialized `.npz`/`.json` detector bundles store `calibration_window_s` and `calibration_hop_s`, and `monitor_stream()` rejects mismatched `window_s` with `ValueError` unless `allow_window_mismatch=True`. Because DCASE clips carry no anomaly-onset timestamps, first-alert timing from clip start is reported as `time_to_first_flag_s_median`.

| Machine | Detector | Calibration Mode (`q=0.99`) | Target Chunk FPR (`1 - q`) | Measured Normal-Chunk FPR (`flagged / total`) | False Alarms / Hour | Median Time to First Flag | Chunk Latency (Median / p95) | Streaming `pAUC` |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `valve` | `gaussian` | `oof_chunk_conformal` | `0.0100` | **`0.0133`** (`12 / 900`, `1.33x`) | `43.2` | `8.5 s` | `0.59 / 0.66 ms` | `0.5084` |
| `valve` | `knn` | `oof_chunk_conformal` | `0.0100` | **`0.0078`** (`7 / 900`, `0.78x`) | `25.2` | `4.0 s` | `0.78 / 0.90 ms` | `0.5084` |

### Cross-Modal Pipetting Cavitation & Stepper-Stall Interlock (`community_extensions/cross_modal_interlock.py`, Issue #37)

`CrossModalInterlockDetector` strips `[300, 3400] Hz` (`raw_audio_saved=False`) and monitors step-synchronized aspirate/dispense windows across low-band stepper stall (`[60, 260] Hz`) and high-band meniscus air-break cavitation (`[3600, 7800] Hz`).

- **Pure-tone caveat**: The `1.0000` fault F1 (`0.0000` false-alarm rate, vs `0.6486` F1 and `1.0000` speech false-alarm rate for `SpeechUnfilteredRMSDetector`) holds **only** for the legacy synthetic pure-tone distractor confined strictly inside `[380, 3150] Hz`, until physical pipetting + real-speech recordings are benchmarked. `RHEOLOGY_PRIORS` are heuristic synthetic simulation constants.
- **Broadband voiced + fricative speech & harmonic rejection**: Real speech includes `85–255 Hz` glottal pitch harmonics (`1/h` up to `7.8 kHz`) and `4–8 kHz` sibilants (`/s/`, `/sh/`). `speech_rejection_mode="harmonic_onset_aware"` uses interharmonic comb-valley floor estimation in `[3600, 7800] Hz`, cross-band `F0` cogging-vs-voice notching in `[60, 260] Hz`, and sustained stroke percentiles (`52` speech-overlapped fault-free clips, `44` active transfers, `48` fault clips):

| Speech Level | Reference | `bandstop_only` False-Halt Rate (All / Active Speech) | `bandstop_only` Fault F1 | `harmonic_onset_aware` False-Halt Rate (All / Active Speech) | `harmonic_onset_aware` Fault F1 |
|---:|---|---:|---:|---:|---:|
| `0 dB` | vs clip RMS | `0.6731` (`0.7955`) | `0.7328` | **`0.0000` (`0.0000`)** | **`0.9787`** |
| `-10 dB` | vs clip RMS | `0.0000` (`0.0000`) | `1.0000` | **`0.0000` (`0.0000`)** | **`1.0000`** |
| `-20 dB` | vs clip RMS | `0.0000` (`0.0000`) | `1.0000` | **`0.0000` (`0.0000`)** | **`1.0000`** |
| `-30 dB` | vs clip RMS | `0.0000` (`0.0000`) | `1.0000` | **`0.0000` (`0.0000`)** | **`1.0000`** |
| `0 dB` | vs room-conversation RMS (`+17.6 dB`) | `0.8462` (`1.0000`) | `0.6857` | `0.4808` (`0.5682`) | `0.6727` |
| `-10 dB` | vs room-conversation RMS (`+7.6 dB`) | `0.8462` (`1.0000`) | `0.6857` | `0.3846` (`0.4545`) | `0.7748` |
| `-20 dB` | vs room-conversation RMS (`-2.4 dB`) | `0.3654` (`0.4318`) | `0.8348` | **`0.0000` (`0.0000`)** | **`1.0000`** |
| `-30 dB` | vs room-conversation RMS (`-12.4 dB`) | `0.0000` (`0.0000`) | `1.0000` | **`0.0000` (`0.0000`)** | **`1.0000`** |

Local 16 kHz real speech WAVs (e.g., LibriSpeech `CC-BY-4.0`) can be mixed directly via `run_benchmark(speech_dir="path/to/wavs")`.

### Privacy & Audio Event Ledger (`acoustic/record.py`)

- Raw audio is not stored (`raw_audio_kept=False`, `raw_audio_saved=False`).
- `--speech-band drop` applies a pre-STFT `[300, 3400] Hz` spectral notch and drops every mel filter whose Hann-skirted support `[lo_hz - 2*df, hi_hz + 2*df]` overlaps `(300, 3400) Hz`, retaining `retained_mel_bands_hz(drop_speech_band=True) == [[0, 255], [3464, 8000]]` (`>= 60 dB` speech-fundamental attenuation).
- A contact microphone or accelerometer mounted directly on the instrument reduces airborne speech pickup by `> 20 dB` compared to a room microphone. Tell people in the room whenever instruments are monitored.

```json
{"schema": "lab-ears/audio-event/v1", "protocol_id": "pcr-setup-v2", "step": 4,
 "instrument_id": "centrifuge-01", "start_utc": "2026-11-18T09:58:12.031Z", "duration_s": 10.0,
 "sample_rate_hz": 16000, "source_sample_rate_hz": 16000, "mic": "contact", "detector": "gaussian",
 "score": 12.5, "threshold": 10.0, "flagged": true, "features_path": null,
 "speech_band_dropped": true, "raw_audio_kept": false, "raw_audio_saved": false,
 "retained_bands_hz": [[0, 255], [3464, 8000]], "notes": ""}
```

## Datasets

| Dataset | Registry ID | Channel / Year | What | License (`commercial_use_ok`) |
|---|---|---|---|---|
| [Wet Lab Protocols (WLP)](https://github.com/chaitanya2334/WLP-Dataset) | *(pinned baseline)* | `protocol_text` (2018) | 622 protocols annotated with actions, reagents, amounts, links | `MIT` (`True`) |
| [LAB-Bench ProtocolQA](https://huggingface.co/datasets/futurehouse/lab-bench) | `lab_bench_protocolqa` | `protocol_text` (2024) | Wet-lab biology protocol troubleshooting & parameter QA | `CC-BY-SA-4.0` (`True`) |
| [ChemBench](https://huggingface.co/datasets/jablonkagroup/ChemBench) | `chembench` | `protocol_text` (2026) | Chemical synthesis, stoichiometry, analytical procedures & lab safety | `MIT` (`True`) |
| [BioProBench](https://huggingface.co/datasets/BioProBench/BioProBench) | `bioprobench` | `protocol_text` (2026) | 22,413 wet-lab protocols across PQA, step ordering, error correction & generation | `CC-BY-NC-4.0` (`False`, **non-commercial**) |
| [BioProt (BioPlanner)](https://github.com/bioplanner/bioplanner) | `bioplanner_bioprot` | `protocol_text` (2024) | `protocols.io` procedures paired with executable planning pseudocode | `NOASSERTION` (`False`, check before use) |
| [ProtoAct (BioP2E)](https://github.com/gjm112233/ProtoAct) | `protoact_biop2e` | `protocol_text` (2026) | Cell-culture SOPs aligned to state-aware robot action sequences | `NOASSERTION` (`False`, check before use) |
| [protocols.io](https://www.protocols.io/) | - | `protocol_text` | Large open protocol repository | Per protocol, often `CC-BY-4.0` |
| [DCASE 2023 Task 2 dev](https://zenodo.org/records/7882613) | *(CPU baseline)* | `machine_audio` (2023) | Normal and anomalous 16 kHz machine sounds, 7 machine types, domain shift | `CC-BY-4.0` (`True`) |
| [DCASE 2024 Task 2 dev](https://zenodo.org/records/10902294) | `dcase2024_task2_dev` | `machine_audio` (2024) | First-shot 16 kHz ASD across 7 machines with hidden operational attributes | `CC-BY-NC-SA-4.0` (`False`, **non-commercial**) |
| [DCASE 2025 Task 2 dev](https://zenodo.org/records/15097779) | `dcase2025_task2_dev` | `machine_audio` (2025) | First-shot 16 kHz 10-12 s ASD across 7 machines + supplemental clean/noise clips | `CC-BY-NC-SA-4.0` (`False`, **non-commercial**) |
| [DCASE 2026 Task 2 dev](https://zenodo.org/records/19336329) | `dcase2026_task2_dev` | `machine_audio` (2026) | Noise-aware two-channel 16 kHz near/far-field machine ASD | `CC-BY-NC-SA-4.0` (`False`, **non-commercial**) |
| [IMAD-DS (2024)](https://zenodo.org/records/12665499) | `imad_ds_2024` | `machine_audio` (2024) | 16 kHz analog mic + 6.7 kHz IMU on wafer-handling robotic arm & brushless motor | `CC-BY-SA-4.0` (`True`) |
| [MIMII](https://zenodo.org/records/3384388) | `mimii` | `machine_audio` (2019) | 8-channel 16 kHz pump, fan, valve & slide-rail sounds at `-6`, `0`, `+6 dB` SNR | `CC-BY-SA-4.0` (`True`) |
| [MIMII DG](https://zenodo.org/records/6529888) | `mimii_dg` | `machine_audio` (2022) | 16 kHz domain-generalization sounds across bearing, fan, gearbox, slider & valve | `CC-BY-NC-SA-4.0` (`False`, **non-commercial**) |
| [ToyADMOS2](https://zenodo.org/records/4580270) | `toyadmos2` | `machine_audio` (2021) | 48 kHz multi-mic miniature-machine operating sounds under domain shift | `LicenseRef-ToyADMOS2-NC` (`False`, **non-commercial**) |
| [LibriSpeech](https://www.openslr.org/12) | - | speech ASR | ~1000 h read English speech for ASR | `CC-BY-4.0` |
| [Common Voice](https://commonvoice.mozilla.org/) | - | speech ASR | Crowd-sourced multilingual speech | `CC0-1.0` |

Licenses marked **non-commercial** (`commercial_use_ok=False`) are fine for research benchmarks but blocked when `--require-commercial-ok` is enabled.

### On-Demand Open Dataset Registry & Loaders (`acoustic/open_datasets.py`)

```bash
python scripts/fetch_open_datasets.py --list --min-year 2024
python scripts/fetch_open_datasets.py --verify-urls
python scripts/fetch_open_datasets.py --dataset lab_bench_protocolqa --require-commercial-ok
python scripts/fetch_open_datasets.py --dataset mimii --machines pump --snr-db 0 --require-commercial-ok
python scripts/fetch_open_datasets.py --dataset dcase2025_task2_dev --machines fan valve
```

## Open-source repos

Stars as of 2026-11-18.

| Repo | What | Stars | License |
|---|---|---:|---|
| [openai/whisper](https://github.com/openai/whisper) | Robust Speech Recognition via Large-Scale Weak Supervision | 112,989 | MIT |
| [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp) | Port of OpenAI's Whisper model in C/C++ | 55,281 | MIT |
| [explosion/spaCy](https://github.com/explosion/spaCy) | Industrial-strength Natural Language Processing (NLP) in Python | 34,068 | MIT |
| [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Faster Whisper transcription with CTranslate2 | 26,845 | MIT |
| [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | Speech-to-text, text-to-speech, speaker diarization, speech enhancement, source... | 16,020 | Apache-2.0 |
| [huggingface/distil-whisper](https://github.com/huggingface/distil-whisper) | Distilled variant of Whisper for speech recognition. 6x faster, 50% smaller, within 1%... | 4,117 | MIT |
| [Future-House/LAB-Bench](https://github.com/Future-House/LAB-Bench) | Evaluation dataset for AI systems intended to benchmark capabilities foundational to... | 136 | CC-BY-SA-4.0 |
| [YuyangSunshine/bioprobench](https://github.com/YuyangSunshine/bioprobench) | Code and data for BioProBench | 73 | custom (see repo) |
| [bioplanner/bioplanner](https://github.com/bioplanner/bioplanner) | Data from the BioPlanner paper | 30 | no license file |
| [chaitanya2334/WLP-Dataset](https://github.com/chaitanya2334/WLP-Dataset) | - | 17 | MIT |

## Papers

| Paper | Venue | Link |
|---|---|---|
| Robust Speech Recognition via Large-Scale Weak Supervision | ICML 2023 | [arXiv:2212.04356](https://arxiv.org/abs/2212.04356) |
| Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling | arXiv 2023 | [arXiv:2311.00430](https://arxiv.org/abs/2311.00430) |
| An Annotated Corpus for Machine Reading of Instructions in Wet Lab Protocols | NAACL 2018 | [arXiv:1805.00195](https://arxiv.org/abs/1805.00195) |
| BioPlanner: Automatic Evaluation of LLMs on Protocol Planning in Biology | EMNLP 2023 | [arXiv:2310.10632](https://arxiv.org/abs/2310.10632) |
| LAB-Bench: Measuring Capabilities of Language Models for Biology Research | arXiv 2024 | [arXiv:2407.10362](https://arxiv.org/abs/2407.10362) |
| BioProBench: A Corpus and Benchmark for Biological Protocol Reasoning in Autonomous Science | ICML 2026 | [arXiv:2505.07889](https://arxiv.org/abs/2505.07889) |

## Conferences and workshops

- [Interspeech 2026](https://www.interspeech2026.org/)
- [ACL 2026](https://2026.aclweb.org/)
- [NeurIPS (main track, Datasets and Benchmarks, workshops)](https://neurips.cc/)
- [AI for Science workshop series (NeurIPS / ICML / ICLR)](https://ai4sciencecommunity.github.io/)
- [SLAS - Society for Laboratory Automation and Screening](https://www.slas.org/)

## Progress

Recent results, newest first. Date = first arXiv version; each line is a short summary of the abstract.

| Date | Paper | In one line |
|---|---|---|
| 2026-10-23 | [BenchBench-Cloning: Evaluating Molecular Cloning Design and Protocol Development](https://arxiv.org/abs/2610.33355) | BenchBench-Cloning: 65 molecular cloning tasks grounded in sequence records, lab notebooks and published studies. |
| 2026-08-24 | [BenchBench-Protocol: Evaluating Real-World Wet-Lab Protocol Reasoning and Modification](https://arxiv.org/abs/2608.23898) | BenchBench-Protocol: 149 protocol-modification tasks recovered from real changes scientists made to published protocols. |
| 2026-08-03 | [ProtoAct: Turning Wet-Lab Protocols into Embodied Robotic Actions](https://arxiv.org/abs/2608.01690) | ProtoAct: turns free-form wet-lab protocols into state-aware robot action sequences. |
| 2025-05-11 | [BioProBench: A Corpus and Benchmark for Biological Protocol Reasoning in Autonomous Science](https://arxiv.org/abs/2505.07889) | BioProBench: 22,413 human-written protocols and 523,784 task instances; models do well on comprehension but drop on deep reasoning, quantities and safety. |

## System Profile: Object · Data · Model · Training · History · Unique Properties

`lab-ears` is part of the [Robot AI Auto-Lab (`auto-lab`)](https://github.com/yaniguan/auto-lab) ecosystem ([Project Studio](https://yaniguan.github.io/auto-lab/)) at **Layer `L2 · Auditory & Protocol Sensory Cortex`**.

- **Object**: Sensory ingestion layer that extracts and normalizes quantitative parameters (`Amount`, `Temperature`, `Time`, `Speed`, plus optional `Concentration`) from wet-lab protocol prose and monitors benchtop machine audio for unsupervised acoustic anomalies and cross-modal pipetting faults with fail-closed speech-band privacy filtering.
- **Data**: Wet Lab Protocols (WLP) 622-protocol BRAT corpus (`374` `train`, `123` `dev`, `125` `test`, MIT, pinned at `1f51ae6e620a5aa3590acf725713f430783f56d7`); DCASE 2023 Task 2 `16 kHz` machine audio (`fan`, `valve`, `bearing`, CC BY 4.0, Zenodo `7882613`); 12 typed open sensory datasets (`OPEN_SENSORY_DATASETS`); and `lab-ears/audio-event/v1` JSONL ledger schema.
- **Model**: Stdlib regex span extractor (`demo/parse_protocol.py`) + zero-leakage Passive-Aggressive (`PA-II`) linear-chain 17-state `BIOES` sequence tagger (`protocol/learned_tagger.py`) with canonical unit normalization (`parse_quantities`) and 1-to-1 bipartite overlap scoring; 64-bin HTK log-mel frontend (pre-STFT `[300, 3400] Hz` notch + Hann-skirt mel exclusion retaining `[[0, 255], [3464, 8000]] Hz`) paired with `GaussianDetector`, `KNNDetector(k=5)`, `AutoencoderDetector`, and `CrossModalInterlockDetector` (`bandstop_only` & `harmonic_onset_aware`).
- **Training**: Zero-leakage structured PA-II training strictly on `WLP/train` with dev-only hyperparameter selection (`0.8761` exact / `0.8974` overlap held-out `test` micro-F1 vs `0.8502` / `0.8798` for regex); one-class normal-only fitting (`1,000` clips/machine) on DCASE 2023 (`0.5617` fullband / `0.5449` speech-dropped best harmonic mean) plus window-matched 20-fold `oof_chunk_conformal` streaming calibration (`0.0133` Gaussian and `0.0078` kNN chunk FPR on `valve` at `q=0.99`).
- **Unique Properties**: Zero external runtime dependencies for `demo/parse_protocol.py` and `protocol/learned_tagger.py` (stdlib fallback), `< 7 ms` median single-thread CPU latency per `10 s` clip (`< 0.8 ms` per `2 s` streaming chunk), non-pickle `.npz`/`.json.gz` model persistence, fail-closed `autolab-wire/v1` privacy checking, and 15 Lean 4.36.0 machine-checked privacy & interlock theorems (`lean/LabEarsLean.lean`).

Full technical specification: [`docs/ARCHITECTURE_AND_HISTORY.md`](docs/ARCHITECTURE_AND_HISTORY.md).

## 15-Repository Auto-Lab Wire Contract (`autolab-wire/v1`)

`autolab_wire.py` exposes `process_packet(packet: dict) -> dict` (`REPO_ID = "lab-ears"`, `LAYER = "L2"`) as the sensory entry node of the 15-repository Auto-Lab closed loop:

- **Upstream / Entry Inputs**: Free-form `goal_text`, default `retained_bands_hz` derived directly from `acoustic.features.retained_mel_bands_hz(drop_speech_band=True)` (`[[0, 255], [3464, 8000]]`), `sample_rate_hz` (`16000`), `window_skirt_hz` (`0.0`), `raw_audio_saved` / `raw_audio_kept` (`False`), and optional `audio_events` (`lab-ears/audio-event/v1`) or `interlock_decision` (`InterlockDecision`). Halted upstream packets (`halted=True`) short-circuit unchanged.
- **Fail-Closed L2 Guardrails (`full_skirt_speech_band_drop`, `cross_modal_cavitation_or_stall_interlock`, `calibrated_anomaly_threshold`)**: Requires every attached `AudioEvent` or `InterlockDecision` record to explicitly declare `speech_band_dropped`, `raw_audio_saved` (or `raw_audio_kept`), and `retained_bands_hz`—halting missing or violating records with `halt_reason = "L2_SPEECH_PRIVACY_BLOCK"` and naming the missing field(s) in `trace[-1]["summary"]`. Propagates active cavitation/stall halts (`"L2_CAVITATION_INTERLOCK_HALT"`) and calibrated anomaly flags (`"L2_ACOUSTIC_ANOMALY_HALT"`). When `defaults_applied` is non-empty, sets trace status to `"WARN"` and emits `"defaults_only": True` whenever `goal_text` contains zero parsed quantities.
- **Downstream Handoff (`lab-ears -> lab-ask-why, stepbot`)**: Emits `parsed_spans`, `defaults_only`, `acoustic_telemetry`, `speech_band_dropped`, `trace`, and `bottom_reasoning_traces` across the 15-repository stack (`lab-ears` `L2` -> `lab-ask-why`, `lab-decide` `L6` -> `lab-automation-atlas`, `buy-or-make`, `labshift` `L5` -> `lab-rehearsal`, `lab-verify` `L4` -> `stepbot`, `lab-hands` `L1` -> `lab-feet` `L0` -> `lab-attrib` `L4` -> `twinstep`, `labfill`, `labbench-posttrain` `L3`).

## Repository Layout

```text
lab-ears/
├── .github/workflows/
│   └── demo.yml                  # CI: stdlib tests, WLP output.txt diff gate, acoustic pytest, Lean 4.36.0 build + axioms
├── acoustic/
│   ├── __init__.py               # Public acoustic anomaly & streaming monitoring exports
│   ├── detect.py                 # Gaussian, kNN, Autoencoder, window-matched OOF chunk calibration & safe bundles
│   ├── evaluate.py               # Single-thread DCASE 2023 Task 2 offline & causal streaming benchmark runner
│   ├── features.py               # Dtype-aware WAV loader, pre-STFT notch + Hann-skirt log-mel, chunk slicer
│   ├── fetch_dcase2023.py        # Timeout + MD5 + Zip-Slip-guarded Zenodo downloader
│   ├── monitor.py                # CLI/API emitting calibrated AudioEvent JSONL lines with window-mismatch guard
│   ├── open_datasets.py          # Typed 2024-2026+ OpenSensoryDataset registry & WAV/JSONL loaders
│   ├── record.py                 # Validated lab-ears/audio-event/v1 ledger schema
│   └── requirements.txt          # numpy, scipy, scikit-learn, threadpoolctl, pytest
├── community_extensions/
│   └── cross_modal_interlock.py  # Voice-stripped cavitation/stall interlock & harmonic comb-valley speech rejection
├── demo/
│   ├── parse_protocol.py         # Zero-dependency regex extractor, unit normalizer & BRAT scorer
│   └── output.txt                # Verified WLP test-split regex metrics (125 protocols)
├── docs/
│   └── ARCHITECTURE_AND_HISTORY.md
├── lean/
│   ├── LabEarsLean.lean          # Lean 4.36.0 voice-privacy, Nyquist, outward-rounding & interlock proofs
│   └── README.md                 # Proof inventory & broadband speech false-halt characterization
├── protocol/
│   ├── __init__.py               # Public sequence-labeling exports
│   └── learned_tagger.py         # Zero-leakage PA-II linear-chain BIOES span tagger & external transfer runner
├── results/
│   ├── dcase2023_cpu.json        # Single-thread CPU DCASE 2023 offline & 2.0 s streaming evaluation
│   └── wlp_learned_vs_rules.json # Per-split & pooled 622-protocol WLP + BioProBench/BioPlanner benchmark
├── scripts/
│   ├── fetch_open_datasets.py    # On-demand MD5 + Zip-Slip-guarded open dataset downloader
│   └── verify_lean_axioms.py     # Lean 4 lake build + #print axioms verifier
├── tests/
│   ├── test_acoustic.py          # Unit tests for acoustic features, detectors, bundles & 2 s streaming FPR
│   ├── test_autolab_wire.py      # Unit tests for fail-closed autolab-wire/v1 privacy gate & WARN defaults
│   ├── test_community_extensions.py
│   ├── test_learned_tagger.py    # Unit & benchmark-drift tests for LinearChainSpanTagger & committed JSONs
│   ├── test_open_datasets_2026.py
│   └── test_parse_protocol.py    # Offline unit tests for protocol span extraction & normalization
├── autolab_wire.py               # L2 autolab-wire/v1 node (protocol spans + fail-closed speech-band gate)
├── LICENSE                       # MIT License
├── pyproject.toml                # Pytest configuration
├── README.md                     # Overview, benchmarks, curated datasets, repos, and papers
└── requirements.txt              # Standard library only for protocol parser & learned tagger
```

## Contributing

Found a broken link or a missing dataset, repo or paper? Open an issue or a pull request.

## License

Code in this repo: MIT. Listed datasets, repos and papers keep their own licenses.
