import json
import os
import shutil
from dotenv import load_dotenv
from mcp import StdioServerParameters
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def load_env_config() -> dict:
    """加载环境变量配置"""
    logger.info("加载环境变量")
    load_dotenv()
    return {
        "API_KEY": os.getenv("API_KEY"),
        "BASE_URL": os.getenv("BASE_URL"),
        "MODEL": os.getenv("MODEL"),
        "MAX_TOKENS": int(os.getenv("MAX_TOKENS", "10000"))
    }

def validate_config(config: dict, required_keys: list):
    """验证配置是否包含必需的键"""
    for key in required_keys:
        if not config.get(key):
            raise ValueError(f"环境变量 {key} 未设置")

def load_mcp_config(config_path: str) -> Dict[str, StdioServerParameters]:
    """加载 MCP 服务器配置文件，返回服务器名称到StdioServerParameters的映射"""
    logger.info(f"加载MCP配置文件: {config_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"配置文件JSON格式错误: {str(e)}")
    except UnicodeDecodeError as e:
        raise ValueError(f"配置文件编码错误: {str(e)}")
    
    if "mcpServers" not in config:
        raise ValueError("配置文件缺少 'mcpServers' 部分")
    
    servers_params = {}
    
    for server_name, server_config in config["mcpServers"].items():
        try:
            if "command" not in server_config:
                logger.warning(f"服务器 '{server_name}' 配置缺少 'command' 字段，跳过")
                continue
            
            command_path = shutil.which(server_config["command"])
            if not command_path:
                logger.warning(f"服务器 '{server_name}' 命令不可用: {server_config['command']}，跳过")
                continue
            
            stdio_params = StdioServerParameters(
                command=command_path,
                args=server_config.get("args", []),
                env=server_config.get("env", None)
            )
            
            servers_params[server_name] = stdio_params
            logger.info(f"成功加载MCP服务器配置: {server_name}")
            logger.info(f"命令路径: {command_path}")
            
        except Exception as e:
            logger.error(f"加载服务器 '{server_name}' 配置失败: {str(e)}")
            continue
    
    if not servers_params:
        raise ValueError("没有找到有效的MCP服务器配置")
    
    return servers_params