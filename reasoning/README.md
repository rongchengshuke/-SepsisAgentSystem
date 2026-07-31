# Reasoning Showcase

这个文件夹用于给“基于LLM的多智能体协同推理”做小样本展示准备。

包含内容：
- `sampled_20_cases.csv`：随机抽取的20例窗口级样本
- `sampled_20_cases.jsonl`：20例的结构化转文本结果
- `cases/`：每例单独一个Markdown，方便人工看数据到文本的转换
- `prompt_template.md`：统一提示词模板
- `reasoning_quality_checklist.md`：人工核查推理质量用
- `model_candidates_top10_2026-07-21.csv/.md`：10个模型候选

当前样本来源：
- 数据集：`data/processed_dataset_outputs_compact_los1_7_year1119`
- 抽样方式：固定随机种子 `20260721`
- 抽样时间：2026-07-21
