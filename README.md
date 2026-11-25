# lab-ears

Teach a lab agent to *listen*: turn spoken goals, written protocols and instrument logs into clear, checkable steps, and notice when a machine sounds wrong.

## Where it fits

In the lab agent on [yaniguan.github.io](https://yaniguan.github.io/), the ears are planned. They take chemistry and biology protocols, a scientist's goal by voice or chat, and instrument logs and alarms, and pass clear steps to the brain (YLM, a model that reads text and 3D molecular geometry together). This repo collects what exists today and starts with the simplest piece: pulling amounts, temperatures, times and speeds out of real protocol text.

## First attempt

A rule-based baseline (regular expressions, no learning) that finds **Amount, Temperature, Time and Speed** spans in wet-lab protocols, normalizes them into canonical Auto-Lab units (`parse_quantities()`), and scores them against the gold labels of the Wet Lab Protocols corpus test split (125 protocols, commit `1f51ae6e620a5aa3590acf725713f430783f56d7`). Overlap metrics (`P_overlap`, `R_overlap`, `F1_overlap`) use 1-to-1 maximum bipartite matching.

```bash
git clone https://github.com/chaitanya2334/WLP-Dataset data/WLP-Dataset
git -C data/WLP-Dataset checkout 1f51ae6e620a5aa3590acf725713f430783f56d7
python demo/parse_protocol.py data/WLP-Dataset/test --min-f1 0.78
```

Output (CPU only, re-measured against pinned WLP test split):

```
protocols: 125 (discontinuous_gold: 2, skipped_discontinuous: 0)
label           gold  pred  P_exact  R_exact F1_exact  P_overlap  R_overlap F1_overlap
Amount          1013  1132    0.756    0.845    0.798      0.777      0.869      0.821
Temperature      509   583    0.799    0.916    0.853      0.851      0.974      0.908
Time             812   801    0.915    0.903    0.909      0.944      0.931      0.937
Speed            155   146    0.863    0.813    0.837      0.932      0.877      0.904
```

Plain rules reach exact F1 0.80-0.91 (overlap F1 0.82-0.94) on these four number-like slots. Actions, reagents and the links between them are not tested yet; that is the next step, with a learned parser. Speech input (for example a small Whisper model) comes after that.

## Machine sounds

Instruments often sound wrong before they fail: a worn bearing grinds, an unbalanced rotor knocks, a pump stalls. `acoustic/` learns what normal sounds like for one machine and flags clips that do not fit. It never sees an anomaly in training.

- Features: 64-bin log-mel spectrogram (16 kHz, 1024-point FFT, hop 512), numpy and scipy only.
- Detectors: `gaussian` (Mahalanobis distance of the clip's per-bin mean and std), `knn` (mean distance to the 5 nearest normal clips), `autoencoder` (dense 128-128-8-128-128 on 5 stacked frames; score is the reconstruction error).
- Data: three machines (fan, valve, bearing) from the [DCASE 2023 Task 2 development set](https://zenodo.org/records/7882613) (CC BY 4.0). Per machine: 1000 normal training clips (990 source domain, 10 target domain), 200 test clips of 10 s. The script downloads them; no audio is in this repo.

```bash
pip install -r acoustic/requirements.txt
python -m acoustic.fetch_dcase2023 --machines fan valve bearing   # about 850 MB
python -m acoustic.evaluate --machines fan valve bearing
python -m pytest -q tests/test_acoustic.py                        # synthetic clips, no download
```

Measured (Linux, Intel Xeon CPU, one thread, 2026-11-18; full output in `results/dcase2023_cpu.json`). AUC per domain and pAUC (max FPR 0.1), as in the DCASE rules:

| Machine | Detector | AUC source | AUC target | pAUC | Fit time | Latency per 10 s clip (median / p95) |
|---|---|---:|---:|---:|---:|---:|
| fan | gaussian | 0.475 | 0.499 | 0.551 | 0.1 s | 5.3 / 6.7 ms |
| fan | knn | 0.402 | 0.461 | 0.517 | 0.1 s | 4.0 / 4.4 ms |
| fan | autoencoder | 0.719 | 0.498 | 0.584 | 19.1 s | 5.7 / 17.9 ms |
| valve | gaussian | 0.686 | 0.527 | 0.502 | 0.1 s | 2.9 / 7.1 ms |
| valve | knn | 0.586 | 0.453 | 0.491 | 0.1 s | 2.9 / 5.4 ms |
| valve | autoencoder | 0.576 | 0.449 | 0.524 | 18.3 s | 3.3 / 4.0 ms |
| bearing | gaussian | 0.689 | 0.543 | 0.542 | 0.1 s | 2.8 / 3.3 ms |
| bearing | knn | 0.662 | 0.512 | 0.523 | 0.1 s | 2.9 / 3.0 ms |
| bearing | autoencoder | 0.706 | 0.514 | 0.533 | 19.8 s | 3.3 / 7.2 ms |

Latency covers reading the WAV file, the log-mel and the score. Harmonic mean of the nine numbers per detector: autoencoder 0.555, gaussian 0.548, knn 0.502.

What this says:

- This is a hard set on purpose: few target-domain clips and unseen anomaly types. Target-domain AUC is near chance (0.45-0.54) for every detector here.
- For scale, the official DCASE 2023 autoencoder baseline (MSE mode, 5 GPU runs) reports AUC source / target / pAUC of 0.802 / 0.362 / 0.590 on fan, 0.554 / 0.507 / 0.512 on valve and 0.659 / 0.558 / 0.504 on bearing ([task page](https://dcase.community/challenge2023/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring)). Our autoencoder is in the same range.
- Scoring is fast enough for live use (under 20 ms per 10 s clip on one CPU thread). Detection quality is the open problem; pretrained audio embeddings with kNN are the next thing to try.
- These are factory and toy machines, not lab instruments. Lab use needs our own normal recordings per instrument.

### Privacy

Lab audio can catch people talking. By default:

- Raw audio is not stored. Only log-mel features (and the score) are kept.
- `--speech-band drop` (or `log_mel(..., drop_speech_band=True)`) applies a pre-STFT `[300, 3400] Hz` spectral notch and removes every mel filter whose Hann-skirted support `[lo_hz - 2*df, hi_hz + 2*df]` overlaps `(300, 3400) Hz`, keeping only sub-speech (`<= 255 Hz`) and high-band (`>= 3464 Hz`) machine energy (`>= 60 dB` speech-fundamental suppression, CPU/synthetic verified). In the historical 2026-11-18 DCASE 2023 CPU baseline run (`results/dcase2023_cpu.json`, center-frequency mask), dropping the speech band shifted the harmonic mean to autoencoder 0.496, gaussian 0.523, knn 0.487.
- A contact microphone or vibration sensor on the instrument picks up much less speech than a room microphone.

Log-mel features are not a guarantee: coarse speech content may still be recoverable from them. Tell people in the room that machines are being recorded.

### Audio event record

`acoustic/record.py` writes one JSON line per scored clip, ready to join the run ledger:

```json
{"schema": "lab-ears/audio-event/v1", "protocol_id": "pcr-setup-v2", "step": 4,
 "instrument_id": "centrifuge-01", "start_utc": "2026-11-18T09:58:12.031Z", "duration_s": 10.0,
 "sample_rate_hz": 16000, "mic": "contact", "detector": "gaussian", "score": 12.5, "threshold": 10.0,
 "flagged": true, "features_path": null, "speech_band_dropped": true, "raw_audio_kept": false, "notes": ""}
```

The timestamp, protocol id and step tie each sound to the step that was running, so a flagged clip can pause the run or explain a failed one later (`python -m acoustic.monitor --train-dir ... --wav ... --out-jsonl ...` calibrates normal thresholds via `calibrate_threshold()` and appends validated JSONL lines). The example values above are illustrative.

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

External WAV archives and protocol JSON/JSONL corpora are fetched on demand into gitignored `data/<dataset_id>/` (with MD5 verification and Zip-Slip protection) and loaded through speech-band-stripped (`[300, 3400] Hz` excised, Nyquist-checked) or `parse_quantities()` adapters:

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

- **Object**: Sensory ingestion layer that extracts and normalizes quantitative parameters (`Amount`, `Temperature`, `Time`, `Speed`, plus optional `Concentration`) from wet-lab protocol prose and monitors benchtop machine audio for unsupervised acoustic anomalies with speech-band privacy filtering.
- **Data**: Wet Lab Protocols (WLP) 125-protocol BRAT test split (MIT, pinned at `1f51ae6e620a5aa3590acf725713f430783f56d7`); DCASE 2023 Task 2 `16 kHz` machine audio (`fan`, `valve`, `bearing`, CC BY 4.0, MD5-verified Zenodo record `7882613`); 12 typed open sensory datasets (`OPEN_SENSORY_DATASETS`, 2024-2026+ DCASE 2024/2025/2026 Task 2, IMAD-DS 2024, MIMII, MIMII DG, ToyADMOS2, LAB-Bench ProtocolQA, ChemBench, BioProBench, BioPlanner, ProtoAct); and `lab-ears/audio-event/v1` JSONL ledger schema.
- **Model**: Stdlib regex span extractor with contextual `Speed`-vs-`Amount` mass disambiguation, canonical unit normalizer (`parse_quantities`), and 1-to-1 bipartite overlap scorer (`demo/parse_protocol.py`); 64-bin HTK log-mel frontend (strict `[lo_hz, hi_hz]` `300–3400 Hz` speech-band filter exclusion) paired with unsquared Mahalanobis Ledoit-Wolf `GaussianDetector`, few-shot-safe `KNNDetector(k=5)`, and 5-frame `AutoencoderDetector` (`128-128-8-128-128`).
- **Training**: Rule-based zero-parameter baseline on WLP (`0.798–0.909` exact F1, `0.821–0.937` overlap F1); strictly one-class normal-only fitting (`1,000` clips/machine) on DCASE 2023 (`0.555` fullband / `0.496` speech-dropped harmonic mean for autoencoder) plus quantile threshold calibration (`calibrate_threshold`).
- **History**: `ba743be` (2026-11-18, `main` WLP protocol parser & curated resources) -> `5476541` (2026-11-18, `acoustic-anomaly` DCASE 2023 acoustic baseline & privacy event schema) -> merged and hardened with full offline test coverage, canonical quantity normalization, and live `acoustic.monitor` JSONL emission.
- **Unique Properties**: Zero external runtime dependencies for `demo/parse_protocol.py`, `< 6 ms` median single-thread CPU latency per `10 s` WAV clip, dtype-exact PCM normalization, Zip-Slip/MD5-verified dataset fetching, and strict finite/timezone/privacy validation in `AudioEvent.validate()`.

Full technical specification: [`docs/ARCHITECTURE_AND_HISTORY.md`](docs/ARCHITECTURE_AND_HISTORY.md).

## 15-Repository Auto-Lab Wire Contract (`autolab-wire/v1`)

`autolab_wire.py` exposes `process_packet(packet: dict) -> dict` (`REPO_ID = "lab-ears"`, `LAYER = "L2"`) as the sensory entry node of the 15-repository Auto-Lab closed loop:

- **Upstream / Entry Inputs**: Free-form `goal_text`, acoustic `retained_bands_hz` (default `[[60, 260], [3600, 7800]]`), `sample_rate_hz` (`16000`), `window_skirt_hz` (`0.0`), `raw_audio_saved` / `raw_audio_kept` (`False`), and optional `audio_events` (`lab-ears/audio-event/v1`) or `interlock_decision` (`InterlockDecision`). Halted upstream packets (`halted=True`) short-circuit unchanged.
- **L2 Guardrails (`full_skirt_speech_band_drop`, `cross_modal_cavitation_or_stall_interlock`, `calibrated_anomaly_threshold`)**: Enforces the Lean-verified speech-privacy, non-empty band, Nyquist (`2 * hi_hz <= sample_rate_hz`), and Hann-skirt invariants (`halt_reason = "L2_SPEECH_PRIVACY_BLOCK"`), and propagates active cavitation/stepper-stall halts (`"L2_CAVITATION_INTERLOCK_HALT"`) and calibrated anomaly flags (`"L2_ACOUSTIC_ANOMALY_HALT"`) with normalized `acoustic_telemetry`.
- **Downstream Handoff (`lab-ears -> lab-ask-why, stepbot`)**: Emits `parsed_spans` (`spans` with `canonical_min`/`canonical_max`, clause-segmented `steps`, `amounts`, `temperatures`, `times`, `speeds`, `concentrations`, primary-anchored `volume_ul`/`temp_c`/`duration_min`, and `defaults_applied`), `acoustic_telemetry`, `speech_band_dropped`, `trace`, and `bottom_reasoning_traces` for downstream epistemic probing in `lab-ask-why`, liquid-handling compilation in `stepbot`, window scheduling in `labshift`, twin simulation in `lab-rehearsal`, causal blame in `lab-attrib`, and DAG construction in `twinstep`.
- **15-Repository Stack**: `lab-ears` (`L2`) -> `lab-ask-why`, `lab-decide` (`L6`) -> `lab-automation-atlas`, `buy-or-make`, `labshift` (`L5`) -> `lab-rehearsal`, `lab-verify` (`L4`) -> `stepbot`, `lab-hands` (`L1`) -> `lab-feet` (`L0`) -> `lab-attrib` (`L4` post-execution causal blame & certified ledger) -> `twinstep`, `labfill`, `labbench-posttrain` (`L3`, feeding verified updates back to `lab-decide` and `lab-feet`).

## Repository Layout

```text
lab-ears/
├── .github/workflows/
│   └── demo.yml                  # CI workflow: offline pytest, cached pinned WLP-Dataset, F1 gate
├── acoustic/
│   ├── __init__.py               # Public acoustic anomaly & monitoring exports
│   ├── detect.py                 # Gaussian, kNN, Autoencoder detectors & calibrate_threshold()
│   ├── evaluate.py               # Single-thread DCASE 2023 Task 2 benchmark runner
│   ├── features.py               # Dtype-aware WAV loader, zero-copy log-mel, speech-band mask
│   ├── fetch_dcase2023.py        # Timeout + MD5 + Zip-Slip-guarded Zenodo downloader
│   ├── monitor.py                # CLI/API emitting calibrated AudioEvent JSONL lines
│   ├── open_datasets.py          # Typed 2024-2026+ OpenSensoryDataset registry & WAV/JSONL loaders
│   ├── record.py                 # Validated lab-ears/audio-event/v1 ledger schema
│   └── requirements.txt          # numpy, scipy, scikit-learn, threadpoolctl, pytest
├── community_extensions/
│   └── cross_modal_interlock.py  # Voice-stripped cavitation & stepper-stall interlock
├── demo/
│   ├── parse_protocol.py         # Zero-dependency regex extractor, unit normalizer & BRAT scorer
│   └── output.txt                # Reference WLP test-split metrics (125 protocols)
├── docs/
│   └── ARCHITECTURE_AND_HISTORY.md
├── lean/
│   └── LabEarsLean.lean          # Lean 4 voice-privacy & cross-modal interlock proofs
├── results/
│   └── dcase2023_cpu.json        # Reference single-thread CPU DCASE 2023 evaluation
├── scripts/
│   └── fetch_open_datasets.py    # On-demand MD5 + Zip-Slip-guarded open dataset downloader
├── tests/
│   ├── test_acoustic.py          # Synthetic unit tests for acoustic features, detectors & ledger
│   ├── test_autolab_wire.py      # Unit tests for autolab-wire/v1 packet processing & privacy block
│   ├── test_community_extensions.py
│   ├── test_open_datasets_2026.py # Registry, DCASE 2024-2026/MIMII WAV & protocol JSONL tests
│   └── test_parse_protocol.py    # Offline unit tests for protocol span extraction & normalization
├── autolab_wire.py               # L2 autolab-wire/v1 node (protocol spans + speech-band gate)
├── LICENSE                       # MIT License
├── pyproject.toml                # Pytest configuration
├── README.md                     # Overview, benchmarks, curated datasets, repos, and papers
└── requirements.txt              # Standard library only for protocol parser
```

## Contributing

Found a broken link or a missing dataset, repo or paper? Open an issue or a pull request.

## License

Code in this repo: MIT. Listed datasets, repos and papers keep their own licenses.
