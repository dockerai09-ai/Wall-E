[English](../../en/knowledge/01-knowledge-module.md) · **简体中文**

[← 返回 README](../../../README.zh-cn.md) · [文档索引](../README.md)

# 知识模块：RAG、知识图谱、评测、治理与溯源

本页的中文翻译尚未完成，请先阅读[英文版](../../en/knowledge/01-knowledge-module.md)。

简要说明：Wall-E 可以基于你自己的文档回答问题并给出引用。检索采用向量 + 关键词（FTS5）混合并做倒数排名融合，再通过受本体约束的知识图谱扩展上下文（可选镜像到 Neo4j）。治理策略（`knowledge/governance.yaml`）在每次请求中强制执行；每个回答都会生成溯源记录（支持 W3C PROV-JSON 导出）；评测工具（`npm run kb:eval`）用于度量检索与回答质量。
