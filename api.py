import asyncio
import json
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from config_loader import load_env_config, validate_config
from chatServiceLanggraph import EnhancedChatService as ChatService # 使用重构后的字节流MCP版本
from uvicorn import Config, Server
from contextlib import asynccontextmanager
import signal

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(f"./logs/api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 定义请求体模型
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户输入的消息")
    session_id: str = Field(default="default", description="会话ID")
    use_mcp: bool = Field(False, description="是否启用MCP服务（已废弃，MCP在初始化时配置）")
    
    class Config:
        # 允许额外字段但忽略它们，保持向后兼容
        extra = "ignore"

class MCPServerConfig(BaseModel):
    name: str = Field(..., description="MCP服务器名称")
    transport: str = Field(default="stdio", description="传输协议：stdio")
    command: str = Field(..., description="stdio模式下的命令")
    args: Optional[List[str]] = Field(default_factory=list, description="命令参数")
    env: Optional[Dict[str, str]] = Field(default_factory=dict, description="环境变量")

    @field_validator("command")
    def validate_stdio_config(cls, v):
        if not v:
            raise ValueError("需要指定command")
        return v

# 全局ChatService实例
chat_service: Optional[ChatService] = None

def load_mcp_config(config_path: str = "./mcp_config.json") -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    mcp_servers = config.get("mcpServers", {})
    return mcp_servers

# 寿命周期事件处理程序
@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理FastAPI的启动和关闭事件"""
    global chat_service
    try:
        env_config = load_env_config()
        validate_config(env_config, ["API_KEY", "BASE_URL", "MODEL"])
        logger.info("环境配置加载成功")

        # 加载MCP配置
        mcp_servers = load_mcp_config()
        
        # 初始化ChatService
        chat_service = ChatService(env_config, mcp_servers)
        
        # 测试连接
        if not await chat_service.test_api_connection():
            logger.error("LLM API连接失败")
            raise HTTPException(status_code=500, detail="LLM API连接失败，请检查配置")

        # 测试MCP连接
        if mcp_servers:
            mcp_results = await chat_service.test_mcp_connections()
            working_servers = sum(1 for status in mcp_results.values() if status)
            logger.info(f"MCP服务器连接测试完成：{working_servers}/{len(mcp_results)} 个正常工作")

        logger.info("聊天服务初始化完成")
        yield

    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务启动失败: {str(e)}")
    finally:
        if chat_service:
            try:
                await chat_service.cleanup()
                logger.info("聊天服务资源已清理")
            except Exception as e:
                logger.error(f"清理聊天服务失败: {str(e)}")

# 创建FastAPI应用
app = FastAPI(
    title="LLM Chat API with MCP Support",
    description="Enhanced API for interacting with LLM and MCP services using stream-based implementation",
    version="2.1.0",
    lifespan=lifespan
)

# 优化CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有
    allow_credentials=False,  # 注意：allow_credentials=True 时不能用 "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_class=StreamingResponse)
async def chat(request: ChatRequest, fastapi_request: Request):
    """处理流式聊天请求，支持MCP功能直接调用"""
    global chat_service
    if not chat_service:
        raise HTTPException(status_code=500, detail="聊天服务未初始化")

    try:
        logger.info(f"收到聊天请求，消息长度: {len(request.message)}")

        async def stream_response():
            response_generated = False
            try:
                async with asyncio.timeout(60):  # 60秒超时
                    async for response_dict in chat_service.stream_message(request.message):
                        if response_dict :
                            response_generated = True
                            yield json.dumps(response_dict, ensure_ascii=False) + "\n"
                        else:
                            logger.debug(f"无效的 response_dict: {response_dict}")

                if not response_generated:
                    logger.warning("没有生成任何响应内容")
                    yield json.dumps({
                        "status": "warning",
                        "message": "没有生成有效的响应内容",
                        "timestamp": datetime.now().isoformat()
                    }, ensure_ascii=False) + "\n"

            except asyncio.TimeoutError:
                logger.error("流式响应超时")
                yield json.dumps({
                    "status": "error",
                    "message": "响应超时",
                    "timestamp": datetime.now().isoformat()
                }, ensure_ascii=False) + "\n"
            except asyncio.CancelledError:
                logger.info("流式响应被客户端取消")
                raise
            except Exception as e:
                logger.error(f"流式响应错误: {str(e)}", exc_info=True)
                yield json.dumps({
                    "status": "error",
                    "message": f"处理请求失败: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }, ensure_ascii=False) + "\n"

        return StreamingResponse(
            stream_response(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        logger.error(f"处理聊天请求失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理请求失败: {str(e)}")

@app.post("/chat/debug")
async def chat_debug(request: ChatRequest):
    """调试用的聊天端点，返回非流式响应"""
    global chat_service
    if not chat_service:
        raise HTTPException(status_code=500, detail="聊天服务未初始化")

    try:
        logger.info(f"调试模式处理消息: {request.message[:50]}...")
        responses = []
        
        async with asyncio.timeout(60):
            async for response_dict in chat_service.stream_message(
                request.message,
            ):
                if response_dict:
                    responses.append(response_dict)

        # 获取MCP功能信息
        mcp_functions = await chat_service.list_available_functions()
        
        return {
            "success": True,
            "message": "调试响应",
            "responses": responses,
            "response_count": len(responses),
            "session_id": request.session_id,
            "mcp_enabled": len(mcp_functions) > 0,
            "mcp_functions": [func["name"] for func in mcp_functions],
            "available_functions_count": len(mcp_functions),
            "implementation": "stream-based"
        }
    except asyncio.TimeoutError:
        logger.error("调试请求超时")
        return {
            "success": False,
            "error": "调试请求超时",
            "session_id": request.session_id,
            "mcp_enabled": False,
            "implementation": "stream-based"
        }
    except Exception as e:
        logger.error(f"调试处理出错: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "session_id": request.session_id,
            "mcp_enabled": False,
            "implementation": "stream-based"
        }

@app.get("/test")
async def test_endpoint():
    """测试API是否正常运行"""
    global chat_service
    if not chat_service:
        return {
            "message": "API服务正常运行",
            "timestamp": datetime.now().isoformat(),
            "mcp_enabled": False,
            "mcp_functions_count": 0,
            "service_status": "not_initialized",
            "implementation": "stream-based"
        }
    
    try:
        # 获取MCP功能信息
        mcp_functions = await chat_service.list_available_functions()
        return {
            "message": "API服务正常运行",
            "timestamp": datetime.now().isoformat(),
            "mcp_enabled": len(mcp_functions) > 0,
            "mcp_functions_count": len(mcp_functions),
            "service_status": "initialized",
            "implementation": "stream-based"
        }
    except Exception as e:
        logger.error(f"获取服务状态失败: {str(e)}", exc_info=True)
        return {
            "message": "API服务正常运行，但获取MCP状态失败",
            "timestamp": datetime.now().isoformat(),
            "mcp_enabled": False,
            "mcp_functions_count": 0,
            "service_status": "initialized_with_errors",
            "error": str(e),
            "implementation": "stream-based"
        }

@app.get("/mcp/status")
async def mcp_status():
    """检查MCP服务状态"""
    global chat_service
    if not chat_service:
        raise HTTPException(status_code=500, detail="聊天服务未初始化")

    try:
        # 测试MCP连接
        mcp_connections = await chat_service.test_mcp_connections()
        mcp_functions = await chat_service.list_available_functions()
        
        # 统计功能类型
        function_types = {}
        for func in mcp_functions:
            func_type = func.get('type', 'unknown')
            function_types[func_type] = function_types.get(func_type, 0) + 1
        
        return {
            "mcp_enabled": len(mcp_functions) > 0,
            "mcp_servers_count": len(chat_service.mcp_server_configs),
            "mcp_functions": mcp_functions,
            "function_types": function_types,
            "connection_status": mcp_connections,
            "working_servers": sum(1 for status in mcp_connections.values() if status),
            "total_servers": len(mcp_connections),
            "timestamp": datetime.now().isoformat(),
            "implementation": "stream-based"
        }
    except Exception as e:
        logger.error(f"获取MCP状态失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取MCP状态失败: {str(e)}")

@app.get("/mcp/functions")
async def list_mcp_functions():
    """列出所有可用的MCP功能"""
    global chat_service
    if not chat_service:
        raise HTTPException(status_code=500, detail="聊天服务未初始化")

    try:
        mcp_functions = await chat_service.list_available_functions()
        
        # 按类型分组
        grouped_functions = {}
        for func in mcp_functions:
            func_type = func.get('type', 'unknown')
            if func_type not in grouped_functions:
                grouped_functions[func_type] = []
            grouped_functions[func_type].append(func)
        
        return {
            "functions": mcp_functions,
            "grouped_functions": grouped_functions,
            "total_count": len(mcp_functions),
            "function_types": list(grouped_functions.keys()),
            "timestamp": datetime.now().isoformat(),
            "implementation": "stream-based"
        }
    except Exception as e:
        logger.error(f"获取MCP功能列表失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取MCP功能列表失败: {str(e)}")

@app.post("/mcp/reload")
async def reload_mcp_config(config_path: str = "./mcp_config.json"):
    """重新加载MCP配置"""
    global chat_service
    if not chat_service:
        raise HTTPException(status_code=500, detail="聊天服务未初始化")

    try:
        await chat_service.reload_mcp_clients()   # 重新创建注册表
        
        # 重新加载配置
        mcp_servers = load_mcp_config(config_path)
        chat_service.mcp_server_configs = mcp_servers
        
        # 重新初始化MCP客户端
        if mcp_servers:
            await chat_service._initialize_mcp_clients()
            
        # 测试连接
        mcp_results = await chat_service.test_mcp_connections()
        working_servers = sum(1 for status in mcp_results.values() if status)
        
        # 获取功能信息
        mcp_functions = await chat_service.list_available_functions()

        return {
            "success": True,
            "message": f"MCP配置重新加载完成",
            "config_path": config_path,
            "loaded_servers": len(mcp_servers),
            "working_servers": working_servers,
            "total_functions": len(mcp_functions),
            "connection_results": mcp_results,
            "implementation": "stream-based"
        }
    except Exception as e:
        logger.error(f"重新加载MCP配置失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"重新加载MCP配置失败: {str(e)}")

@app.get("/sessions")
async def list_sessions():
    """列出所有活跃的会话"""
    global chat_service
    if not chat_service:
        raise HTTPException(status_code=500, detail="聊天服务未初始化")

    return {
        "sessions": list(chat_service.store.keys()),
        "total_count": len(chat_service.store),
        "timestamp": datetime.now().isoformat()
    }

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """清空指定会话的历史记录"""
    global chat_service
    if not chat_service:
        raise HTTPException(status_code=500, detail="聊天服务未初始化")

    if session_id in chat_service.store:
        chat_service.store[session_id].clear()
        return {
            "success": True,
            "message": f"会话 {session_id} 的历史记录已清空",
            "session_id": session_id
        }
    else:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")

async def run_server():
    """运行FastAPI服务"""
    config = Config(
        app=app, 
        host="0.0.0.0", 
        port=8110,
        log_level="info"
    )
    server = Server(config)

    loop = asyncio.get_event_loop()
    shutdown_event = asyncio.Event()

    async def handle_shutdown():
        if shutdown_event.is_set():
            return
        logger.info("收到中断信号，开始关闭...")
        shutdown_event.set()
        try:
            global chat_service
            if chat_service:
                try:
                    await chat_service.cleanup()
                    logger.info("聊天服务资源已清理")
                except Exception as e:
                    logger.error(f"清理聊天服务失败: {str(e)}")

            # 调用 server.shutdown()，它会优雅停止
            await server.shutdown()
            logger.info("服务器已完全关闭")
        except Exception as e:
            logger.error(f"关闭过程中出错: {str(e)}", exc_info=True)
        
    def signal_handler(_sig, _frame):
        if not shutdown_event.is_set():
            loop.call_soon_threadsafe(lambda: asyncio.create_task(handle_shutdown()))

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal_handler)

    try:
        logger.info("启动API服务器，监听端口: 8110")
        await server.serve()
    except Exception as e:
        logger.error(f"服务器运行时出错: {str(e)}", exc_info=True)
        raise
    finally:
        await handle_shutdown()  # 确保清理

if __name__ == "__main__":
    logger.info("启动增强型API服务 - 字节流MCP实现")
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("程序被用户中断，完成清理")
    except Exception as e:
        logger.error(f"程序启动失败: {str(e)}", exc_info=True)