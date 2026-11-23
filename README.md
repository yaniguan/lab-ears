# lab-ears

Teach a lab agent to *listen*: turn spoken goals, written protocols and instrument logs into clear, checkable steps, and notice when a machine sounds wrong.

## Where it fits

In the lab agent on [yaniguan.github.io](https://yaniguan.github.io/), the ears are planned. They take chemistry and biology protocols, a scientist's goal by voice or chat, and instrument logs and alarms, and pass clear steps to the brain (YLM, a model that reads text and 3D molecular geometry together). This repo collects what exists today and starts with the simplest piece: pulling amounts, temperatures, times and speeds out of real protocol text.

## First attempt

A rule-based baseline (regular expressions, no learning) that finds **Amount, Temperature, Time and Speed** spans in wet-lab protocols, scored against the gold labels of the Wet Lab Protocols corpus test split (125 protocols). `P_overlap` counts a predicted span as right if it overlaps a gold span of the same type.

```bash
git clone --depth 1 https://github.com/chaitanya2334/WLP-Dataset data/WLP-Dataset
python demo/parse_protocol.py data/WLP-Dataset/test
```

Output (Linux, CPU only, 2026-11-18):

```
protocols: 125
label         gold  pred  P_exact  R_exact F1_exact  P_overlap
Amount        1011  1103    0.756    0.825    0.789      0.773
Temperature    509   584    0.724    0.831    0.774      0.839
Time           810   806    0.909    0.905    0.907      0.940
Speed          155   182    0.692    0.813    0.748      0.747
```

Plain rules already get F1 0.75-0.91 on these four number-like slots. Actions, reagents and the links between them are not tested yet; that is the next step, with a learned parser. Speech input (for example a small Whisper model) comes after that.

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
- `--speech-band drop` (or `log_mel(..., drop_speech_band=True)`) removes mel bins centred between 300 and 3400 Hz, where most speech energy is, keeping only the machine band. With the band dropped, the harmonic mean becomes autoencoder 0.496, gaussian 0.523, knn 0.487 (same file), so it costs some accuracy.
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

The timestamp, protocol id and step tie each sound to the step that was running, so a flagged clip can pause the run or explain a failed one later. The example values are made up.

## Datasets

| Dataset | What | License |
|---|---|---|
| [Wet Lab Protocols (WLP)](https://github.com/chaitanya2334/WLP-Dataset) | 622 protocols annotated with actions, reagents, amounts, links | MIT |
| [BioProt (BioPlanner)](https://github.com/bioplanner/bioplanner) | Protocols as pseudocode for planning evaluation | No license file - check before use |
| [BioProBench](https://huggingface.co/datasets/BioProBench/BioProBench) | Protocol QA, step ordering, error correction, generation | CC BY-NC 4.0 (**non-commercial**) |
| [LAB-Bench](https://huggingface.co/datasets/futurehouse/lab-bench) | Biology research tasks incl. ProtocolQA | CC BY-SA 4.0 |
| [protocols.io](https://www.protocols.io/) | Large open protocol repository | Per protocol, often CC BY - check each |
| [DCASE 2023 Task 2 dev](https://zenodo.org/records/7882613) | Normal and anomalous machine sounds, 7 machine types, domain shift | CC BY 4.0 |
| [MIMII](https://zenodo.org/records/3384388) | Pump, fan, valve, slide-rail sounds at three noise levels (7-11 GB per file) | CC BY-SA 4.0 |
| [LibriSpeech](https://www.openslr.org/12) | ~1000 h read English speech for ASR | CC BY 4.0 |
| [Common Voice](https://commonvoice.mozilla.org/) | Crowd-sourced multilingual speech | CC0 |

Licenses marked **non-commercial** are fine for research but not for products.

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

- **Object**: Sensory ingestion layer that extracts quantitative parameters (`Amount`, `Temperature`, `Time`, `Speed`) from wet-lab protocol prose (`main`) and monitors benchtop machine audio for unsupervised acoustic anomalies with speech-band privacy filtering (`origin/acoustic-anomaly`).
- **Data**: Wet Lab Protocols (WLP) 125-protocol BRAT test split (MIT) on `main`; DCASE 2023 Task 2 `16 kHz` machine audio (`fan`, `valve`, `bearing`, CC BY 4.0) and `lab-ears/audio-event/v1` JSONL ledger schema on `origin/acoustic-anomaly`.
- **Model**: Stdlib regex span extractor with `Speed`-over-`Amount` span disambiguation (`demo/parse_protocol.py`); 64-bin HTK log-mel frontend (`300–3400 Hz` speech-band mask) paired with Ledoit-Wolf `GaussianDetector`, `KNNDetector(k=5)`, and 5-frame `AutoencoderDetector` (`128-128-8-128-128`).
- **Training**: Rule-based zero-parameter baseline on WLP (`0.748–0.907` exact F1); strictly one-class normal-only fitting (`1,000` clips/machine) on DCASE 2023 (`0.555` fullband / `0.496` speech-dropped harmonic mean for autoencoder).
- **History**: `ba743be` (2026-11-18, `main` WLP protocol parser & curated resources) -> `5476541` (2026-11-18, `origin/acoustic-anomaly` DCASE 2023 acoustic baseline & privacy event schema).
- **Unique Properties**: Zero external dependencies on `main`, `< 6 ms` median single-thread CPU latency per `10 s` WAV clip, and enforced speech-band/raw-audio privacy validation in `AudioEvent.validate()`.

Full technical specification: [`docs/ARCHITECTURE_AND_HISTORY.md`](docs/ARCHITECTURE_AND_HISTORY.md).

## Repository Layout

```text
lab-ears/
├── .github/workflows/
│   └── demo.yml                  # CI workflow cloning WLP-Dataset and running protocol parser
├── demo/
│   ├── parse_protocol.py         # Zero-dependency regex extractor & WLP BRAT evaluator
│   └── output.txt                # Reference WLP test-split metrics (125 protocols)
├── docs/
│   └── ARCHITECTURE_AND_HISTORY.md
├── LICENSE                       # MIT License
├── README.md                     # Overview, benchmarks, curated datasets, repos, and papers
└── requirements.txt              # Standard library only on main (acoustic/ added in PR branch)
```

## Contributing

Found a broken link or a missing dataset, repo or paper? Open an issue or a pull request.

## License

Code in this repo: MIT. Listed datasets, repos and papers keep their own licenses.
