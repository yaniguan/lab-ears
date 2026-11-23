# lab-ears

Teach a lab agent to *listen*: turn spoken goals, written protocols and instrument logs into clear, checkable steps.

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

## Datasets

| Dataset | What | License |
|---|---|---|
| [Wet Lab Protocols (WLP)](https://github.com/chaitanya2334/WLP-Dataset) | 622 protocols annotated with actions, reagents, amounts, links | MIT |
| [BioProt (BioPlanner)](https://github.com/bioplanner/bioplanner) | Protocols as pseudocode for planning evaluation | No license file - check before use |
| [BioProBench](https://huggingface.co/datasets/BioProBench/BioProBench) | Protocol QA, step ordering, error correction, generation | CC BY-NC 4.0 (**non-commercial**) |
| [LAB-Bench](https://huggingface.co/datasets/futurehouse/lab-bench) | Biology research tasks incl. ProtocolQA | CC BY-SA 4.0 |
| [protocols.io](https://www.protocols.io/) | Large open protocol repository | Per protocol, often CC BY - check each |
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
