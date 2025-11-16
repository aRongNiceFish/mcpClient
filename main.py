import asyncio
import sys
import logging
from datetime import datetime
from config_loader import load_env_config, validate_config
from chat_service import ChatService

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f"./logs/main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

async def main():
    """程序主入口"""
    try:
        # 加载环境变量配置
        env_config = load_env_config()
        validate_config(env_config, ["API_KEY", "BASE_URL", "MODEL"])

        # 解析命令行参数
        use_mcp = False
        mcp_config_path = None
        if '--mcp' in [arg.lower() for arg in sys.argv]:
            use_mcp = True
            mcp_config_path = "./mcp_config.json"

        # 初始化聊天服务
        print("🚀 正在启动聊天服务...")
        chat_service = ChatService(env_config, use_mcp, mcp_config_path)

        # 测试 API 连接
        if not await chat_service.test_api_connection():
            print("❌ API连接失败，请检查配置后重试")
            return

        # 连接 MCP 服务（如果启用）
        if use_mcp and mcp_config_path:
            if not await chat_service.connect_mcp():
                print("❌ MCP服务连接失败，禁用MCP")
                chat_service.use_mcp = False

        # 启动聊天循环
        await chat_service.chat_loop()

    except Exception as e:
        print(f"❌ 程序错误: {str(e)}")
        logger.error(f"主程序异常: {str(e)}")
    finally:
        if 'chat_service' in locals():
            await chat_service.cleanup()

if __name__ == "__main__":
    logger.info("启动程序")
    asyncio.run(main())