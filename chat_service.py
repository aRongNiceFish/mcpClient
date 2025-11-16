import asyncio
import json
import logging
from typing import Optional, List, Dict,AsyncGenerator
from openai.types.chat import ChatCompletionChunk
from openai import OpenAI
from mcp_service import MCPService
from config_loader import load_mcp_config

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, env_config: dict, use_mcp: bool = False, mcp_config_path: Optional[str] = None):
        self.api_key = env_config["API_KEY"]
        self.base_url = env_config["BASE_URL"]
        self.model_name = env_config["MODEL"]
        self.max_tokens = env_config["MAX_TOKENS"]
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # MCP 服务控制
        self.use_mcp = use_mcp
        self.mcp_config_path: Optional[str] = mcp_config_path
        self.mcp_services: Dict[str, MCPService] = {}

        # 打印配置信息
        print("=== 聊天服务配置 ===")
        print(f"API密钥: {'已设置' if self.api_key else '未设置'}")
        print(f"基础URL: {self.base_url}")
        print(f"模型名称: {self.model_name}")
        print(f"最大令牌数: {self.max_tokens}")
        print(f"MCP服务: {'启用' if self.use_mcp else '禁用'}")
        if self.use_mcp and mcp_config_path:
            print(f"MCP配置文件: {mcp_config_path}")
        print("==================")
        
        logger.info("聊天服务初始化完成")

    async def test_api_connection(self) -> bool:
        """测试 LLM API 连接"""
        logger.info("测试 LLM API 连接")
        print("正在测试API连接...")
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            print("✓ API连接成功!")
            logger.info(f"API测试成功，响应: {response.choices[0].message.content}")
            return True
        except Exception as e:
            error_msg = str(e)
            print(f"✗ API连接失败: {error_msg}")
            logger.error(f"API连接测试失败: {error_msg}")
            return False

    async def connect_mcp(self) -> bool:
        """连接 MCP 服务配置中定义的所有服务"""
        if not self.use_mcp or not self.mcp_config_path:
            return False

        try:
            # 加载配置，直接获取 StdioServerParameters 字典
            servers_params = load_mcp_config(self.mcp_config_path)
            
            if not servers_params:
                logger.warning("配置文件中未找到任何有效的 MCP 服务")
                return False

            self.mcp_services.clear()

            for name, stdio_params in servers_params.items():
                try:
                    service = MCPService(stdio_params)
                    if await service.connect():
                        self.mcp_services[name] = service
                        logger.info(f"成功连接 MCP 服务 [{name}]")
                    else:
                        logger.warning(f"MCP 服务 [{name}] 连接失败")
                except Exception as e:
                    logger.error(f"连接 MCP 服务 [{name}] 失败: {str(e)}")

            return len(self.mcp_services) > 0

        except Exception as e:
            logger.error(f"加载 MCP 配置失败: {str(e)}")
            return False

    async def toggle_mcp(self, enable: bool, config_path="./mcp_config.json") -> None:
        """动态启用或禁用 MCP 服务"""
        logger.info(f"切换MCP服务状态: {'启用' if enable else '禁用'}")
        if enable:
            if not config_path:
                print("❌ 请提供MCP配置文件路径")
                logger.error("未提供MCP配置文件路径")
                return
            self.mcp_config_path = config_path
            self.use_mcp = True
            success = await self.connect_mcp()
            if success:
                print("✓ MCP服务已启用")
            else:
                print("❌ MCP服务启用失败")
                self.use_mcp = False
                self.mcp_services.clear()
        else:
            for svc in self.mcp_services.values():
                await svc.disconnect()
            self.mcp_services.clear()
            self.use_mcp = False
            print("✓ MCP服务已禁用")
        logger.info(f"MCP服务状态: {'启用' if self.use_mcp else '禁用'}")

    async def call_tool(self, tool_name: str, tool_args: Dict) -> str:
        """在所有MCP服务中查找并调用支持该工具的服务"""
        for name, svc in self.mcp_services.items():
            if svc.is_connected:
                tools = await svc.list_tools()
                if any(t["function"]["name"] == tool_name for t in tools):
                    logger.info(f"使用服务 [{name}] 调用工具 [{tool_name}]")
                    return await svc.call_tool(tool_name, tool_args)
        raise RuntimeError(f"未找到支持工具 {tool_name} 的 MCP 服务")

    async def process_query(self, query: str) -> str:
        """处理用户查询"""
        logger.info(f"处理查询: {query[:50]}...")
        messages = [{"role": "user", "content": query}]
        available_tools = []

        if self.use_mcp:
            for svc in self.mcp_services.values():
                tools = await svc.list_tools()
                available_tools.extend(tools)

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model_name,
                max_tokens=self.max_tokens,
                messages=messages,
                tools=available_tools if available_tools else None
            )
            message = response.choices[0].message
            result_parts = []

            if message.content:
                result_parts.append(message.content)
                logger.info(f"收到AI回复: {message.content[:100]}...")

            if message.tool_calls and self.use_mcp:
                logger.info(f"执行 {len(message.tool_calls)} 个工具调用")
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        logger.error(f"工具参数JSON解析失败: {tool_call.function.arguments}")
                        continue

                    result_parts.append(f"\n[正在调用工具: {tool_name}]")
                    try:
                        tool_content = await self.call_tool(tool_name, tool_args)
                        messages.append({
                            "role": "assistant",
                            "content": message.content,
                            "tool_calls": [{
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": tool_call.function.arguments
                                }
                            }]
                        })
                        messages.append({
                            "role": "tool",
                            "content": tool_content,
                            "tool_call_id": tool_call.id
                        })
                        final_response = await asyncio.to_thread(
                            self.client.chat.completions.create,
                            model=self.model_name,
                            max_tokens=self.max_tokens,
                            messages=messages
                        )
                        final_content = final_response.choices[0].message.content
                        if final_content:
                            result_parts.append(f"\n{final_content}")
                    except Exception as e:
                        error_msg = f"工具 {tool_name} 执行失败: {str(e)}"
                        logger.error(error_msg)
                        result_parts.append(f"\n[错误: {error_msg}]")

            final_result = "".join(result_parts)
            logger.info("查询处理完成")
            return final_result
        except Exception as e:
            error_msg = f"查询处理失败: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("\n🤖 聊天服务已启动!")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 '!mcp on ' 启用MCP服务")
        print("输入 '!mcp off' 禁用MCP服务")
        print("-" * 50)

        while True:
            try:
                query = input("\n💬 您的问题: ").strip()

                if query.lower() in ['quit', 'exit']:
                    print("👋 再见!")
                    break

                if not query:
                    continue

                if query.startswith('!mcp'):
                    parts = query.split()
                    if len(parts) >= 2:
                        if parts[1].lower() == 'on':
                            await self.toggle_mcp(True)
                        elif parts[1].lower() == 'off':
                            await self.toggle_mcp(False)
                        else:
                            print("❌ 无效的MCP命令。示例: !mcp on 或 !mcp off")
                    continue

                print("\n🔄 正在处理您的问题...")
                response = await self.process_query(query)
                print(f"\n🤖 回答:\n{response}")
                print("-" * 50)

                logger.info("用户查询处理完成")

            except KeyboardInterrupt:
                print("\n👋 程序被用户中断，再见!")
                break
            except Exception as e:
                error_msg = f"处理过程中发生错误: {str(e)}"
                print(f"\n❌ {error_msg}")
                logger.error(error_msg)
                
    async def stream_message(self, message: str, history: List[str]) -> AsyncGenerator[str, None]:
        """
        流式输出用户消息的回复内容，用于 WebSocket 场景。
        """
        logger.info(f"开始流式处理消息: {message[:50]}...")
        messages = []
        for i in range(0, len(history), 2):
            messages.append({"role": "user", "content": history[i]})
            if i + 1 < len(history):
                messages.append({"role": "assistant", "content": history[i + 1]})
        messages.append({"role": "user", "content": message})

        try:
            # 创建流式响应
            response = await asyncio.to_thread(
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    stream=True,
                    max_tokens=self.max_tokens
                )
            )

            for chunk in response:
                if isinstance(chunk, ChatCompletionChunk):
                    delta = chunk.choices[0].delta
                    content = delta.content or ""
                    if content:
                        yield content
        except Exception as e:
            logger.error(f"流式消息处理失败: {str(e)}")
            raise

    async def cleanup(self):
        """清理资源"""
        logger.info("开始清理聊天服务资源")
        for svc in self.mcp_services.values():
            await svc.disconnect()
        logger.info("聊天服务资源清理完成")
