import asyncio
import logging
from typing import AsyncGenerator, Dict, Any, List, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
# 导入 langchain_mcp_adapters 和 mcp
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json
MCP_ADAPTERS_AVAILABLE = True

logger = logging.getLogger(__name__)

class State(TypedDict):
    messages: Annotated[list, add_messages]

class EnhancedChatService:
    """使用 langchain_mcp_adapters 和 mcp 的增强聊天服务"""
    
    def __init__(self, env_config: dict, mcp_servers: Dict = None):
        if not MCP_ADAPTERS_AVAILABLE:
            logger.info("不使用mcp工具")
        
        self.api_key = env_config["API_KEY"]
        self.base_url = env_config["BASE_URL"]
        self.model_name = env_config["MODEL"]
        self.max_tokens = env_config["MAX_TOKENS"]
        self.graph_builder = StateGraph(State)
        # MCP 相关属性
        self.mcp_server_configs = mcp_servers
        self.langchain_tools: List = []
        self.config = {"configurable": {"thread_id": "1"}}
        # 初始化 LangChain ChatOpenAI 客户端
        self.graph = None
        self.llm = None
        self.agent = None  # 添加 agent 属性
        self._initialized = False
        logger.info(f"EnhancedChatService 初始化完成，模型: {self.model_name}")
        if self.mcp_server_configs:
            logger.info(f"配置了 {len(self.mcp_server_configs)} 个 MCP 服务器配置")

    async def _initialize_mcp_clients(self):
        """使用官方 langchain_mcp_adapters API 初始化 MCP 客户端"""
        if self._initialized:
            return

        logger.info("开始使用 langchain_mcp_adapters API 初始化...")
        
        client = MultiServerMCPClient(self.mcp_server_configs)

        # 创建一个持久化的连接管理器
        try:
            self.langchain_tools = await client.get_tools()
            logger.info(f"加载了 {len(self.langchain_tools)} 个工具:")

        except Exception as e:
            logger.error(f"初始化 MCP 客户端失败: {e}", exc_info=True)

        self._initialize_llm()
        # 绑定工具到 LLM
        if self.llm and self.langchain_tools:
            self.llm = self.llm.bind_tools(self.langchain_tools)
        self._initialized = True
        logger.info(f"总共加载了 {len(self.langchain_tools)} 个工具")
        await self.build_graph()
    
    def _initialize_llm(self):
        """初始化 LangChain LLM 和 Agent"""
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model_name,
            max_tokens=self.max_tokens
        )
        # 创建系统提示模板
        self.system_message = "你是一个智能助手，可以回答各种问题并提供帮助。你必须以纯 JSON 格式回答，不要包含任何解释、Markdown 或额外文本。JSON 必须以 { 开头，以 } 结尾。"
    
    async def chatbot(self, state: State):
        # 使用 agent 来处理消息
        result = await self.llm.ainvoke(state["messages"])
        return {"messages" : [result]}
 
    async def build_graph(self):
        
        self.graph_builder.add_node("chatbot", self.chatbot)
        self.tool_node = ToolNode(tools=self.langchain_tools)
        self.graph_builder.add_node("tools", self.tool_node)
        self.graph_builder.add_conditional_edges("chatbot", tools_condition)
        self.graph_builder.add_edge("tools", "chatbot")
        self.graph_builder.add_edge(START, "chatbot")
        memory = InMemorySaver()
        self.graph = self.graph_builder.compile(checkpointer=memory)

    async def stream_message(self, message: str) -> AsyncGenerator[Dict[str, Any], None]:
        """流式输出，自动区分：文本、JSON、工具结果"""
        logger.info(f"开始流式处理消息: {message[:50]}...")

        try:
            async for event in self.graph.astream(
                {"messages": [{"role": "user", "content": message}]},
                self.config,
                stream_mode="values",
            ):
                last_msg: AnyMessage = event["messages"][-1]
                logger.debug(f"收到消息: {type(last_msg).__name__}")

                # ---------- 1. 工具执行结果 (ToolMessage) ----------
                if isinstance(last_msg, ToolMessage):
                    content = last_msg.content

                    # 尝试解析工具返回的 JSON
                    if isinstance(content, str):
                        stripped = content.strip()
                        if stripped.startswith("{") and stripped.endswith("}"):
                            try:
                                parsed = json.loads(stripped)
                                yield {
                                    "type": "json",           # 结构化数据
                                    "data": parsed,
                                    "source": "tool"
                                }
                                continue
                            except json.JSONDecodeError:
                                pass  # 不是 JSON，继续当文本

                    # 工具返回纯文本
                    yield {
                        "type": "tool_result",
                        "name": getattr(last_msg, "name", "unknown"),
                        "content": content
                    }
                    continue
                # ---------- 2. AI 回复 (AIMessage) ----------
                if isinstance(last_msg, AIMessage):
                    raw = last_msg.content
                    # 兼容 OpenAI 格式: list[dict]
                    if isinstance(raw, list):
                        raw = "".join(
                            block.get("text", "") if isinstance(block, dict) else str(block)
                            for block in raw
                        )
                    if not isinstance(raw, str):
                        raw = str(raw)
                    stripped = raw.strip()
                    # 尝试识别 AI 返回的 JSON（有时会直接返回 JSON 字符串）
                    if stripped.startswith("{") and stripped.endswith("}"):
                        try:
                            parsed = json.loads(stripped)
                            yield {
                                "type": "json",
                                "data": parsed,
                                "source": "ai"
                            }
                            continue
                        except json.JSONDecodeError:
                            pass
                    # 普通文本（Markdown）
                    if stripped:
                        yield {
                            "type": "text",
                            "content": stripped
                        }

        except Exception as e:
            logger.error(f"流式处理消息时出错: {e}", exc_info=True)
            yield {
                "type": "error",
                "content": f"处理消息时出错: {str(e)}"
            }

    async def list_available_tools(self) -> List[Dict[str, Any]]:
        """列出所有可用的工具"""
        await self._initialize_mcp_clients()
        tools_info = []
        for tool in self.langchain_tools:
            tools_info.append({
                'name': tool.name,
                'description': tool.description,
                'type': 'tool'
            })
        return tools_info

    async def list_available_functions(self) -> List[Dict[str, Any]]:
        """列出所有可用的功能（兼容性方法）"""
        return await self.list_available_tools()

    async def test_api_connection(self) -> bool:
        """测试 LLM API 连接"""
        logger.info("测试 LLM API 连接")
        try:
            test_llm = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model_name,
                max_tokens=self.max_tokens
            )
            messages = [HumanMessage(content="Hi")]
            response = await asyncio.to_thread(test_llm.invoke, messages)
            logger.info(f"API 测试成功，响应: {response.content[:50]}...")
            return True
        except Exception as e:
            logger.error(f"API 连接测试失败: {e}")
            return False

    async def test_mcp_connections(self) -> Dict[str, bool]:
        """测试所有 MCP 服务器连接"""
        results = {}
        
        await self._initialize_mcp_clients()

        for name, config in self.mcp_server_configs.items():
            server_name = name
            try:
                server_params = StdioServerParameters(
                    command=config.get('command'),
                    args=config.get('args', []),
                    env=config.get('env', {}),
                    transport=config.get('transport', 'stdio')
                )
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        results[server_name] = True
                        logger.info(f"MCP 连接测试成功: {server_name}")
            except Exception as e:
                logger.error(f"MCP 连接测试失败 ({server_name}): {e}")
                results[server_name] = False
        return results

    async def cleanup(self):
        """清理聊天服务资源"""
        logger.info("开始清理聊天服务资源")
        
        # 清理 MCP 连接
        for connection in self.mcp_connections:
            try:
                name = connection.get('name', 'Unknown')
                
                # 清理 session
                if 'session_manager' in connection and connection['session_manager']:
                    await connection['session_manager'].__aexit__(None, None, None)
                
                # 清理 stdio connection
                if 'stdio_connection' in connection and connection['stdio_connection']:
                    await connection['stdio_connection'].__aexit__(None, None, None)
                
                logger.info(f"已清理 MCP 连接: {name}")
            except Exception as e:
                logger.error(f"清理 MCP 连接失败: {e}")
        
        self.mcp_connections.clear()
        self.langchain_tools.clear()
        self.agent = None
        self._initialized = False
        logger.info("聊天服务资源清理完成")

    async def reload_mcp_clients(self):
        """重新加载 MCP 客户端"""
        logger.info("重新加载 MCP 客户端...")
        await self.cleanup()
        await self._initialize_mcp_clients()
        logger.info("MCP 客户端重新加载完成")