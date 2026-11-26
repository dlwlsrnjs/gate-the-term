# Data and model provenance

The compressed assets contain exactly the inputs used by this camera-ready protocol. SHA256s, uncompressed destinations, and sizes are in `data/assets_manifest.json`. `prepare.py --data-only` verifies them before use.

| Resource | Frozen version and provenance |
|---|---|
| MGSM | Four original TSVs: Bengali, Telugu, Thai, English. Bytes are checked against `scripts/data_checksums.json`. Original release: [Google Research MGSM](https://github.com/google-research/url-nlp/tree/main/mgsm), dataset description: [MGSM](https://huggingface.co/datasets/juletxara/mgsm). |
| Indic-GSM8K | [AI4Bharat release](https://huggingface.co/datasets/ai4bharat/indic-gsm8k), revision `c6899fb49e5405d5a5e339adebe78895b70e8b42`. Its dataset metadata specifies Apache-2.0. Input Parquet SHA256: `71e4759386e6b6ab2c936b0a52cbfb825153f5fc78d0916fb34a232274d656c0`. We retain Bengali/Telugu questions and numeric answers for 1,069 source items after excluding all 250 MGSM items. |
| MSVAMP | [MathOctopus official repository](https://github.com/microsoft/MathOctopus), commit `4d385db696c420b212ccf861fcfab13c1499a576`. The upstream repository supplies an MIT license. The released English/Bengali/Thai files are aligned JSONL, despite their `.json` suffix. The actual release has no Telugu file. We retain all 1,000 source items in Bengali and Thai. |
| Model | [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), revision `a09a35458c702b33eeacc393d103063234e8bc28`. Weights are downloaded separately from the model publisher. Metadata checksums are in `data/model_metadata.json`. |

Upstream raw URLs and file SHA256s are in `data/source_downloads.json`; `prepare.py --download-raw` obtains those exact files. `rebuild_evaluation.py` rebuilds the selected JSONL files. English normalization uses Unicode NFKC, casefold, and alphanumeric characters. All 250 MGSM English questions match the Indic source. No normalized exact overlap was found between the two external selections. This does not exclude paraphrases, shared templates, or pretraining exposure.

External resources are evaluation-only. English source questions are retained for provenance and overlap checks, and never sent to the model at evaluation. Dataset ownership and original terms remain with their publishers; this package adds no new human translation annotations.

For publication, cite the original MGSM, GSM8K/Indic release, and MathOctopus/MSVAMP sources. Report the two external benchmarks separately; their languages and translation provenance differ.
