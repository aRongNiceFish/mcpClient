# app.py
import asyncio
import json
import logging
import os
import signal
from typing import Dict, Any, List
from flask import stream_with_context
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

# ------------------- 配置与工具 -------------------
from config_loader import load_env_config, validate_config
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ------------------- 日志 -------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("./logs/flask_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------- Flask 应用 -------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ------------------- 静态文件目录 -------------------
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)   # 自动创建 static 文件夹

# ------------------- 全局服务 -------------------
chat_service = None

# ------------------- LangGraph State -------------------
class State(TypedDict):
    messages: List

# ------------------- EnhancedChatService -------------------
class EnhancedChatService:
    def __init__(self, env_config: dict, mcp_servers: Dict = None):
        self.api_key = env_config["API_KEY"]
        self.base_url = env_config["BASE_URL"]
        self.model_name = env_config["MODEL"]
        self.max_tokens = env_config.get("MAX_TOKENS", 4096)
        self.mcp_server_configs = mcp_servers or {}
        self.langchain_tools = []
        self.graph = None
        self.llm = None
        self._initialized = False

    async def _initialize_mcp_clients(self):
        if self._initialized:
            return
        logger.info("正在初始化 MCP 客户端...")
        client = MultiServerMCPClient(self.mcp_server_configs)
        try:
            self.langchain_tools = await client.get_tools()
            logger.info(f"成功加载 {len(self.langchain_tools)} 个 MCP 工具")
        except Exception as e:
            logger.error(f"MCP 初始化失败: {e}")

        self.llm = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model_name,
            max_tokens=self.max_tokens
        )
        if self.langchain_tools:
            self.llm = self.llm.bind_tools(self.langchain_tools)

        await self.build_graph()
        self._initialized = True

    async def build_graph(self):
        builder = StateGraph(State)
        builder.add_node("chatbot", self.chatbot)
        builder.add_node("tools", ToolNode(self.langchain_tools))
        builder.add_conditional_edges("chatbot", tools_condition)
        builder.add_edge("tools", "chatbot")
        builder.add_edge(START, "chatbot")
        self.graph = builder.compile(checkpointer=InMemorySaver())

    async def chatbot(self, state: State):
        result = await self.llm.ainvoke(state["messages"])
        return {"messages": [result]}

    async def stream_message(self, message: str, thread_id: str = "default"):
        await self._initialize_mcp_clients()
        config = {"configurable": {"thread_id": thread_id}}
        async for event in self.graph.astream(
            {"messages": [HumanMessage(content=message)]},
            config,
            stream_mode="values"
        ):
            msg = event["messages"][-1]

            # 工具调用结果
            if getattr(msg, "type", None) == "tool":
                yield {
                    "type": "tool_result",
                    "name": getattr(msg, "name", "unknown"),
                    "content": msg.content
                }
                continue

            # AI 文本回复
            if hasattr(msg, "content"):
                content = msg.content
                if isinstance(content, list):
                    content = "".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in content
                    )
                if isinstance(content, str) and content.strip():
                    yield {"type": "text", "content": content.strip()}

    async def list_available_functions(self):
        await self._initialize_mcp_clients()
        return [
            {"name": t.name, "description": t.description}
            for t in self.langchain_tools
        ]

    async def test_api_connection(self) -> bool:
        try:
            test_llm = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model_name,
                max_tokens=self.max_tokens
            )
            resp = await asyncio.to_thread(test_llm.invoke, [HumanMessage(content="ping")])
            return bool(resp.content)
        except Exception as e:
            logger.error(f"LLM 连接测试失败: {e}")
            return False

    async def test_mcp_connections(self) -> Dict[str, bool]:
        results = {}
        for name, cfg in self.mcp_server_configs.items():
            try:
                params = StdioServerParameters(
                    command=cfg.get("command"),
                    args=cfg.get("args", []),
                    env=cfg.get("env", {})
                )
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        results[name] = True
            except Exception as e:
                logger.error(f"MCP {name} 连接失败: {e}")
                results[name] = False
        return results

    async def cleanup(self):
        self.langchain_tools.clear()
        self._initialized = False
        logger.info("EnhancedChatService 已清理")

    async def reload_mcp_clients(self):
        await self.cleanup()
        await self._initialize_mcp_clients()

# ------------------- Flask 路由 -------------------

# 1. 提供前端页面
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(STATIC_DIR, "retro-chat.html")

@app.route("/retro-chat.html", methods=["GET"])
def retro_chat():
    return send_from_directory(STATIC_DIR, "retro-chat.html")

# 2. 聊天流式接口
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")
    if not message:
        return jsonify({"error": "message 不能为空"}), 400

    def generate():
        async def stream():
            async for chunk in chat_service.stream_message(message, session_id):
                yield json.dumps(chunk, ensure_ascii=False) + "\n"
        # 正确运行 async generator
        return asyncio.run(stream())

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")
# 3. 健康检查
@app.route("/test", methods=["GET"])
def test():
    return jsonify({
        "message": "Flask API 运行正常",
        "mcp_enabled": len(chat_service.mcp_server_configs) > 0 if chat_service else False,
        "implementation": "flask"
    })

# 4. 列出 MCP 函数
@app.route("/mcp/functions", methods=["GET"])
def mcp_functions():
    funcs = asyncio.run(chat_service.list_available_functions())
    return jsonify(funcs)

# 5. 热重载 MCP
@app.route("/mcp/reload", methods=["POST"])
def reload_mcp():
    asyncio.run(chat_service.reload_mcp_clients())
    return jsonify({"success": True, "message": "MCP 配置已重新加载"})

# 6. MCP 状态
@app.route("/mcp/status", methods=["GET"])
def mcp_status():
    conn = asyncio.run(chat_service.test_mcp_connections())
    funcs = asyncio.run(chat_service.list_available_functions())
    return jsonify({
        "working_servers": sum(conn.values()),
        "total_servers": len(conn),
        "functions_count": len(funcs),
        "connection_status": conn
    })

# ------------------- 辅助函数 -------------------
def load_mcp_config(path: str = "mcp_config.json") -> Dict:
    if not os.path.exists(path):
        logger.warning(f"{path} 不存在，使用空配置")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("mcpServers", {})
    except Exception as e:
        logger.error(f"加载 {path} 失败: {e}")
        return {}

# ------------------- 优雅关闭 -------------------
def shutdown_handler():
    logger.info("收到关闭信号，正在清理...")
    if chat_service:
        asyncio.run(chat_service.cleanup())
    logger.info("服务已安全关闭")

signal.signal(signal.SIGINT, lambda s, f: shutdown_handler())
signal.signal(signal.SIGTERM, lambda s, f: shutdown_handler())

# ------------------- 主入口 -------------------
if __name__ == "__main__":
    try:
        env_config = load_env_config()
        validate_config(env_config, ["API_KEY", "BASE_URL", "MODEL"])
        mcp_servers = load_mcp_config()

        chat_service
        chat_service = EnhancedChatService(env_config, mcp_servers)

        logger.info("Flask 服务启动 → http://127.0.0.1:8110")
        app.run(host="0.0.0.0", port=8110, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"启动失败: {e}")
        raise