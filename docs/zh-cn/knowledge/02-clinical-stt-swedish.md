[English](../../en/knowledge/02-clinical-stt-swedish.md) · **简体中文**

[← 返回 README](../../../README.zh-cn.md) · [文档索引](../README.md)

# 临床语音转写（瑞典语）：Wall-E 与 Claude Pro 的对比、基准测试与优化循环

本页的中文翻译尚未完成，请先阅读[英文版](../../en/knowledge/02-clinical-stt-swedish.md)。

简要说明：Wall-E 在现有 `/v1/audio/transcriptions` 之上增加了面向临床场景的转写管线（术语表引导识别、受治理的 LLM 纠错、瑞典个人身份号码 personnummer 脱敏、可选的病历字段结构化、仅存哈希的溯源记录），提供 `npm run stt:bench` 基准测试（WER、CER、术语召回率、PII 泄漏率、延迟）以及 `npm run stt:loop` 优化循环，目标是在同一基准上超过手动采集的 Claude Pro 转写结果。
