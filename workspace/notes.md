# 课程笔记：AI 智能体工程

## 什么是 Agent
Agent = LLM（推理决策）+ Tools（改变外部世界）+ Memory（跨轮次状态）+ Controller（控制流）。
缺 Tools 或 Controller 的，通常只是工作流，不是 Agent。

## 为什么需要 MCP
MCP（Model Context Protocol）把"工具实现"和"模型调用"解耦：
工具以独立进程提供，Agent 作为客户端连接，两边用 JSON-RPC 2.0 通信。
好处：换模型不用改工具，换工具不用改模型。

## 生产级与 demo 的分界线
幂等、超时、重试、限流、审计日志、成本上限、灰度回滚、评测集回归。
八项缺一项，就还是 demo。

## 待办
- 完成 MCP 握手流程的时序图
- 补充对抗样本的评测用例
