# 10个候选模型（2026-07-21）

> 说明：前两位仍以 DeepSeek 为主，第 3/4 位改为其他国产系列模型 GLM-5.2 和 ERNIE-5.1，用来做跨系列对照。

| Rank | Provider | Model ID | Status | 推荐用途 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 1 | DeepSeek | `deepseek-v4-pro` | current | 主力高质量推理模型，优先用于20例展示与多轮对比。 | DeepSeek V4 正式旗舰，适合做临床长文本和结构化证据综合推理。 |
| 2 | DeepSeek | `deepseek-v4-flash` | current | 低成本快速批量生成推理文本，适合首轮筛选。 | 同属 V4 系列，速度更快，便于大样本跑通。 |
| 3 | Zhipu | `glm-5.2` | current | 国产高能力对照组，适合做结构化病情推理对比。 | 智谱官方 GLM-5.2 系列，可作为非 DeepSeek 路线的主力备选。 |
| 4 | Baidu | `ERNIE-5.1` | current | 国产中文文本对照组，适合病历摘要和风险解释生成。 | 文心大模型 5.1 系列，可作为另一条国产系列基线。 |
| 5 | Qwen | `qwen3.8-max-preview` | preview_token_plan_only | 如果要冲更高上限，可作为高配对照。 | 预览版，适合少量高质量实验。 |
| 6 | Qwen | `qwen3.7-max` | current | Qwen 高能力主力候选，适合和 DeepSeek V4-Pro 对照。 | 适合正式文本推理实验。 |
| 7 | Qwen | `qwen3.7-plus` | current | 能力与成本平衡，适合 20 例正式比较。 | 作为中档稳定选择比较合适。 |
| 8 | Qwen | `qwen3.6-plus` | current | 平衡档稳定基线，适合批量推理文本生成。 | 更适合预算受限时铺量测试。 |
| 9 | Qwen | `qwen3.6-flash` | current | 低成本快速跑样本，适合首轮粗筛模型。 | 速度优先，便于做 prompt 初筛。 |
| 10 | Qwen | `qwen3-max` | current | Qwen 正式 Max 备选，可和 3.7-max 做版本对照。 | 适合做系列内对比。 |

## 官方来源
- DeepSeek V4 发布页：https://api-docs.deepseek.com/news/news260424/
- DeepSeek 当前模型列表：https://api-docs.deepseek.com/api/list-models/
- Zhipu GLM-5.2 官方页：https://open.bigmodel.cn/dev/api/normal-model/glm-5_2
- Baidu ERNIE-5.1 官方页：https://cloud.baidu.com/doc/WENXINWORKSHOP/s/6mvek1y19
- Qwen 模型总览：https://help.aliyun.com/zh/model-studio/models
- Qwen 文本生成模型页：https://help.aliyun.com/zh/model-studio/text-generation-model/