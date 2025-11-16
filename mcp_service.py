import asyncio
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from utils import serialize_content
import logging

logger = logging.getLogger(__name__)

class MCPService:
    def __init__(self, config: StdioServerParameters):
        self.config = config
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.is_connected = False

    async def connect(self) -> bool:
        """连接到 MCP 服务器"""
        logger.info("连接到MCP服务器")
        print("正在连接到MCP服务器...")
        
        try:
            stdio_transport = await asyncio.wait_for(
                self.exit_stack.enter_async_context(stdio_client(self.config)),
                timeout=60.0
            )
            self.stdio, self.client_transport = stdio_transport
            self.session = await asyncio.wait_for(
                self.exit_stack.enter_async_context(ClientSession(self.stdio, self.client_transport)),
                timeout=60.0
            )
            await asyncio.wait_for(self.session.initialize(), timeout=60.0)
            response = await asyncio.wait_for(self.session.list_tools(), timeout=60.0)
            tools = response.tools
            self.is_connected = True
            print(f"✓ 成功连接到MCP服务器!")
            print(f"可用工具: {[tool.name for tool in tools]}")
            logger.info(f"MCP服务器连接成功，工具列表: {[tool.name for tool in tools]}")
            return True
        except asyncio.TimeoutError:
            error_msg = "连接MCP服务器超时"
            print(f"✗ {error_msg}")
            logger.error(error_msg)
            return False
        except Exception as e:
            error_msg = f"连接MCP服务器失败: {str(e)}"
            print(f"✗ {error_msg}")
            logger.error(error_msg)
            return False

    async def disconnect(self):
        """断开与 MCP 服务器的连接并清理资源"""
        logger.info("开始断开MCP服务器连接")
        self.is_connected = False
        try:
            await self.exit_stack.aclose()
            self.session = None
            logger.info("MCP服务器资源清理完成")
            print("✓ MCP服务器已断开")
        except Exception as e:
            if isinstance(e, ProcessLookupError):
                logger.info("进程已终止，跳过清理")
            else:
                logger.error(f"MCP服务器资源清理失败: {str(e)}")
                print(f"✗ MCP服务器断开失败: {str(e)}")

    async def list_tools(self) -> List[Dict]:
        """获取可用工具列表"""
        if not self.is_connected or not self.session:
            logger.warning("MCP服务器未连接，无法列出工具")
            return []
        try:
            response = await asyncio.wait_for(self.session.list_tools(), timeout=30.0)
            return [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            } for tool in response.tools]
        except Exception as e:
            logger.error(f"获取工具列表失败: {str(e)}")
            return []

    async def call_tool(self, tool_name: str, tool_args: Dict) -> str:
        """调用指定工具"""
        if not self.is_connected or not self.session:
            raise RuntimeError("MCP服务器未连接")
        logger.info(f"调用工具: {tool_name}，参数: {tool_args}")
        try:
            tool_result = await asyncio.wait_for(
                self.session.call_tool(tool_name, tool_args),
                timeout=120.0
            )
            tool_content = serialize_content(tool_result.content)
            logger.info(f"工具 {tool_name} 执行成功，序列化内容: {tool_content}")
            return tool_content
        except asyncio.TimeoutError:
            error_msg = f"工具 {tool_name} 执行超时"
            logger.error(error_msg)
            raise TimeoutError(error_msg)
        except Exception as e:
            error_msg = f"工具 {tool_name} 执行失败: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)