# Project Structure

## Top Level

```text
program/
|- configs/                  runtime yaml config
|- data/                     processed dataset artifacts
|  |- processed/             triples, labels, vocab, sample index
|  `- reasoning/             generated reasoning text and test json
|- docs/                     human-readable project notes
|- knowledge_base/           local RAG markdown documents
|- models/                   model assets and checkpoints
|  |- checkpoints/
|  |  `- trained_pipeline/   end-to-end training output `.pt` files
|  |- encoders/
|  |  |- multimodal_encoder/ figure-1 module-1 weight slot
|  |  |- reasoning_encoder/  reasoning branch weight slot
|  |  |- static_encoder/     static branch weight slot
|  |  `- text_encoder/
|  |     `- bioclinicalbert/ local Hugging Face cache
|  `- fusion_heads/          optional exported fusion/head weights
|- outputs/                  generated prediction tables
|- scripts/
|  |- framework/             framework source code
|  |  |- data/               preprocessing and dataset modules
|  |  |- models/             encoder and fusion implementations
|  |  |- reasoning/          retriever, context builder, LangGraph flow
|  |  `- utils/              shared utilities
|  |- data/                  data-building entry points
|  |- train/                 training entry points
|  |- infer/                 inference entry points
|  `- tools/                 helper entry points
`- vendor/                   upstream reference repositories
```

## What To Open First

If you want to understand the pipeline quickly:

1. `configs/default.yaml`
2. `scripts/data/build_triples.py`
3. `scripts/train/train.py`
4. `scripts/infer/predict.py`
5. `scripts/framework/models/carer_tamf.py`
6. `scripts/framework/reasoning/langgraph_pipeline.py`

## Where Different Files Belong

- runnable entry points: `scripts/data/`, `scripts/train/`, `scripts/infer/`, `scripts/tools/`
- new Python framework code: `scripts/framework/models/`
- dataset builder and dataset classes: `scripts/framework/data/`
- RAG and reasoning library modules: `scripts/framework/reasoning/`
- shared helpers: `scripts/framework/utils/`
- downloaded encoder weights: `models/encoders/`
- train output checkpoints: `models/checkpoints/trained_pipeline/`
- generated prediction tables: `outputs/`
- local RAG knowledge snippets: `knowledge_base/docs/`
- one-off validation outputs: `data/reasoning/`

## About `scripts/framework`

`scripts/framework/` now contains the entire framework implementation:

- module 1 style structured EHR encoder
- reasoning/text encoder branch
- fusion and task heads
- knowledge-base retrieval
- LangGraph and DeepSeek integration helpers

The directories beside it under `scripts/` are now only command entry points.
