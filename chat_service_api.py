import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, Any, List, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing_extensions import TypedDict

# 导入 langchain_mcp_adapters 和 mcp
try:
    from langchain_mcp_adapters.tools import load_mcp_tools
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_ADAPTERS_AVAILABLE = True
except ImportError:
    print("警告: langchain_mcp_adapters 和 mcp 未安装，请运行: pip install langchain-mcp-adapters mcp")
    MCP_ADAPTERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnhancedChatService:
    """使用 langchain_mcp_adapters 和 mcp 的增强聊天服务"""
    
    def __init__(self, env_config: dict, mcp_servers: Dict = None):
        if not MCP_ADAPTERS_AVAILABLE:
            raise ImportError("langchain_mcp_adapters 和 mcp 未安装。请运行: pip install langchain-mcp-adapters mcp")
        
        self.api_key = env_config["API_KEY"]
        self.base_url = env_config["BASE_URL"]
        self.model_name = env_config["MODEL"]
        self.max_tokens = env_config["MAX_TOKENS"]
        self.store = {}

        # MCP 相关属性
        self.mcp_server_configs = mcp_servers
        self.mcp_connections: List[Dict[str, Any]] = []  # 存储持久化连接
        self.langchain_tools: List = []

        # 初始化 LangChain ChatOpenAI 客户端
        self.llm = None
        self.agent = None  # 添加 agent 属性
        self._initialized = False
        self.checkpointer = MemorySaver()
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
        self._initialized = True
        logger.info(f"总共加载了 {len(self.langchain_tools)} 个工具")
    
    def _initialize_llm(self):
        """初始化 LangChain LLM 和 Agent"""
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model_name,
            max_tokens=self.max_tokens
        )
        # 创建系统提示模板
        self.system_message = "你是一个智能助手，可以回答各种问题并提供帮助。"
        
        # 创建 React Agent - 不使用 state_modifier 参数
        self.agent = create_react_agent(
            model=self.llm, 
            tools=self.langchain_tools,
            prompt=self.system_message,
            checkpointer=self.checkpointer
            )
        self._initialized = True
        logger.info(f"已创建 React Agent, 准备处理消息")

    async def stream_message(self, message: str, session_id: str = "default") -> AsyncGenerator[Dict[str, Any], None]:
        """流式输出用户消息的回复内容，支持原生工具调用"""
        logger.info(f"开始流式处理消息: {message[:50]}...")
        #这里不能把id固定，后期想办法在不同的聊天下产生不同的id
        config = {"configurable": {"thread_id": "1"}}
        try:
            await self._initialize_mcp_clients()
            
            if not self.agent:
                raise ValueError("Agent 未初始化")
            
            yield {"type": "info", "content": "正在生成响应...\n"}
            
            # 使用字典格式调用 agent
            input_data = {"messages": message}
            final_messages = ""
            # 流式处理 agent 响应
            async for chunk in self.agent.astream(input_data, config=config):
                # 处理不同类型的输出块
                if 'tools' in chunk and 'messages' in chunk['tools']:
                    for tool_message in chunk['tools']['messages']:
                        if hasattr(tool_message,'content') and tool_message.content:
                            final_messages += tool_message.content
                if 'agent' in chunk and 'messages' in chunk['agent']:
                    for message in chunk['agent']['messages']:
                        if hasattr(message, 'content') and message.content:  # hasattr检查属性存在，message.content直接访问
                            final_messages += message.content
                            print(f"提取的总结消息: {message.content}")

            logger.info("流式消息处理完成")

        except Exception as e:
            logger.error(f"流式消息处理失败: {e}", exc_info=True)
            yield {"type": "error", "content": f"流式消息处理失败: {str(e)}"}

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