"""统一异常分层：决定"能重试 / 要人工 / 直接终止"三种处理路径（C4）。

- RetryableError：网络抖动、限流、5xx -> 可退避重试
- UserInputError：参数不合法 -> 终止当前工具调用，把原因回喂给模型让它自己改正
- FatalError：配置缺失、沙箱越界、权限拒绝 -> 终止本轮，绝不重试
"""


class AgentError(Exception):
    """所有业务异常的基类。"""
    code = "agent_error"


class RetryableError(AgentError):
    code = "retryable"


class UserInputError(AgentError):
    """参数错误：消息会回传给模型，模型可自行修正参数后重试。"""
    code = "bad_request"


class FatalError(AgentError):
    code = "fatal"


class PermissionDenied(FatalError):
    code = "permission_denied"


class ToolTimeout(RetryableError):
    code = "tool_timeout"


class LlmUnavailable(RetryableError):
    code = "llm_unavailable"
