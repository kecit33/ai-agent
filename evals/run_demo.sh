#!/usr/bin/env bash
# DeepSeek 真实模型演示脚本
# 用法：在 mcp-agent 项目根目录执行
#   bash evals/run_demo.sh
# 结果写入 evals/demo_results.md
# 注意：脚本读取本机 .env 中的 LLM_API_KEY，Key 不会外发。

cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1
export LLM_MODE=openai

# 按你的实际 Python 路径调整（虚拟环境里的 python 均可）
P="${PYTHON:-C:/Users/kecit/.workbuddy/binaries/python/envs/default/Scripts/python.exe}"
R=evals/demo_results.md

{
  echo "# DeepSeek 真实模型演示结果"
  echo
  echo "- 模型：deepseek-chat（base_url=https://api.deepseek.com/v1）"
  echo "- 运行时间：$(date '+%Y-%m-%d %H:%M')"
  echo "- 运行方式：CLI --once，会话 ID 相同的条目共享上下文记忆"
  echo
} > "$R"

run() {
  echo "## $1" >> "$R"; echo >> "$R"
  echo "**用户**：$3" >> "$R"; echo >> "$R"
  echo '```' >> "$R"
  "$P" -m src.agent.cli --session "$2" --once "$3" 2>&1 \
    | grep -v '"level"' | grep -v '^$' >> "$R"
  echo '```' >> "$R"; echo >> "$R"
}

run_c() {
  echo "## $1" >> "$R"; echo >> "$R"
  echo "**用户**：$3（人工确认环节输入：$4）" >> "$R"; echo >> "$R"
  echo '```' >> "$R"
  echo "$4" | "$P" -m src.agent.cli --session "$2" --once "$3" 2>&1 \
    | grep -v '"level"' | grep -v '^$' >> "$R"
  echo '```' >> "$R"; echo >> "$R"
}

run   "演示 1：添加待办（单步工具调用）"            101 "帮我添加待办：周五晚上八点前提交小组作业"
run   "演示 2：同一会话追加待办"                    101 "再加一条待办：周日复习 408 操作系统第 3 章"
run   "演示 3：列出待办"                            101 "列出我的待办"
run   "演示 4：完成第一条待办（多轮指代消解）"      101 "把第一条待办标记为完成"
run   "演示 5：天气查询（外部 API）"                102 "查一下北京今天和明天的天气"
run   "演示 6：多轮上下文（那上海呢）"              102 "那上海呢？"
run_c "演示 7：写文件需人工确认（输入 y 允许）"     102 "根据北京的天气写一份三日出行建议，保存到 workspace/travel.md" "y"
run   "演示 8：列出目录文件"                        103 "看看 workspace 目录里有什么文件"
run   "演示 9：读文件并总结"                        103 "读一下 workspace/notes.md 的内容并帮我总结"
run   "演示 10：统计分析 CSV"                       103 "统计一下 workspace/sales_demo.csv"
run   "演示 11：时间查询"                           104 "现在几点了？今天星期几？"
run_c "演示 12：危险操作人工否决（输入 n 拒绝）"    104 "把密码 123456 保存到 workspace/password.txt" "n"

echo "ALL_DONE -> $R"
