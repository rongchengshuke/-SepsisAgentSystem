# Elder Sepsis CARER-TAMF Prototype

A Sepsis Agent System for early recognition of septic shock.

This project is a modular prototype for elderly sepsis adverse-outcome modeling. It combines:

- TAMF-style multi-source time-series encoding for laboratory, vital-sign, and treatment data.
- Static structured features.
- LLM or multi-agent clinical reasoning text.
- Figure-1-style fusion and prediction heads for `death_12`, `death_24`, and `death_48`.

The latest handoff document is:

```text
项目交接文档_2026-07-31.md
```

## Layout

```text
configs/                 Runtime configuration.
data/                    Processed triples, reasoning inputs, notes, and memory files.
knowledge_base/          Local knowledge-base text. RAG is currently disabled.
log/                     Current DeepSeek single-LLM and multi-agent example outputs.
models/                  Encoder caches, exported weights, and trained checkpoints.
scripts/data/            Data preprocessing entry points.
scripts/framework/       Core Figure-1 framework code.
scripts/train/           Stage-1 TAMF and end-to-end training entry points.
scripts/infer/           Prediction entry point.
scripts/tools/           LLM and multi-agent experiment tools.
vendor/                  Upstream reference repositories.
修改下/                  Change records.
```

## Main Config

```text
configs/dataset_compact_los1_7_year1119.yaml
```

Important current settings:

- RAG is disabled.
- Image input is disabled.
- Raw admission notes are used by LLM/agents, not directly encoded by the Figure-1 model.
- Reasoning text is encoded by BioClinicalBERT.
- The current binary tasks are `death_12`, `death_24`, and `death_48`.

## Common Commands

Run commands from the project root with an activated Python/Conda environment.

Rebuild LLM-imputed dynamic/static inputs:

```powershell
python scripts\data\build_llm_imputed_dynamic.py `
  --config configs/dataset_compact_los1_7_year1119.yaml `
  --mode original_flow
```

Train Stage-1 TAMF encoder:

```powershell
python scripts\train\train_stage1_tamf_split.py `
  --config configs/dataset_compact_los1_7_year1119.yaml `
  --save-dir vendor/TAMF/deep_learning/checkpoints/stage1_compact_local `
  --epochs 10 `
  --batch-size 32 `
  --num-workers 4 `
  --pin-memory `
  --embed-dim 128 `
  --num-heads 4
```

Train the Figure-1 model:

```powershell
python scripts\train\train.py `
  --config configs/dataset_compact_los1_7_year1119.yaml `
  --epochs 2 `
  --batch-size 8 `
  --checkpoint-path models/checkpoints/trained_pipeline/latest.pt
```

Run prediction:

```powershell
python scripts\infer\predict.py `
  --config configs/dataset_compact_los1_7_year1119.yaml `
  --checkpoint-path models/checkpoints/trained_pipeline/latest.pt `
  --split test `
  --output-csv outputs/predictions.csv
```

Run DeepSeek single-LLM examples:

```powershell
$env:DEEPSEEK_API_KEY='your_api_key'
$env:PYTHONIOENCODING='utf-8'
python scripts\tools\run_nine_model_rag_eval.py `
  --config configs/dataset_compact_los1_7_year1119.yaml `
  --samples-jsonl reasoning/sampled_20_cases.jsonl `
  --max-samples 5 `
  --providers DeepSeek `
  --model-ids deepseek-v4-flash `
  --log-dir log/deepseek_llm_5_original_preprocess
```

Run DeepSeek multi-agent examples:

```powershell
$env:DEEPSEEK_API_KEY='your_api_key'
$env:PYTHONIOENCODING='utf-8'
python scripts\tools\run_multiagent_flash_compare.py `
  --config configs/dataset_compact_los1_7_year1119.yaml `
  --samples-jsonl reasoning/sampled_20_cases.jsonl `
  --max-samples 5 `
  --include-deepseek `
  --skip-glm `
  --disable-memory `
  --max-consult-rounds 1 `
  --request-note-limit 2400 `
  --log-dir log/deepseek_multiagent_5_original_preprocess_no_memory
```
