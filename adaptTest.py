import asyncio
import json
import logging
from typing import Optional, AsyncGenerator, Dict, Any, List
from langchain_core.tools import BaseTool
# 导入 langchain_mcp_adapters 并进行调试
try:
    from langchain_mcp_adapters.tools import load_mcp_tools
    MCP_ADAPTERS_AVAILABLE = True
    print("✅ langchain_mcp_adapters 导入成功")
except ImportError as e:
    print(f"❌ langchain_mcp_adapters 导入失败: {e}")
    MCP_ADAPTERS_AVAILABLE = False

logger = logging.getLogger(__name__)

async def debug_load_mcp_tools():
    """调试 load_mcp_tools 函数的行为"""
    if not MCP_ADAPTERS_AVAILABLE:
        print("❌ langchain_mcp_adapters 不可用")
        return
    
    # 测试配置
    test_configs = [
        {
            "name": "test-server",
            "command": "echo",  # 使用简单的 echo 命令进行测试
            "args": ["test"]
        }
    ]
    
    for config in test_configs:
        print(f"\n🔍 测试配置: {config}")
        
        try:
            # 尝试不同的调用方式
            print("📝 尝试调用方式 1: 直接传递配置字典")
            result1 = await load_mcp_tools(config)
            print(f"  结果类型: {type(result1)}")
            print(f"  结果内容: {result1}")
            
            if isinstance(result1, dict):
                print("  字典键:", list(result1.keys()))
                for key, value in result1.items():
                    print(f"    {key}: {type(value)} - {value}")
            elif isinstance(result1, list):
                print(f"  列表长度: {len(result1)}")
                for i, item in enumerate(result1):
                    print(f"    [{i}]: {type(item)} - {item}")
            
        except Exception as e:
            print(f"  ❌ 方式 1 失败: {e}")
        
        try:
            print("\n📝 尝试调用方式 2: 使用关键字参数")
            result2 = await load_mcp_tools(
                command=config["command"],
                args=config["args"]
            )
            print(f"  结果类型: {type(result2)}")
            print(f"  结果内容: {result2}")
            
        except Exception as e:
            print(f"  ❌ 方式 2 失败: {e}")
        
        try:
            print("\n📝 尝试调用方式 3: 传递 server_config 参数")
            result3 = await load_mcp_tools(server_config=config)
            print(f"  结果类型: {type(result3)}")
            print(f"  结果内容: {result3}")
            
        except Exception as e:
            print(f"  ❌ 方式 3 失败: {e}")

# 备选方案：检查 langchain_mcp_adapters 的源码结构
def inspect_mcp_adapters():
    """检查 langchain_mcp_adapters 的结构"""
    if not MCP_ADAPTERS_AVAILABLE:
        return
        
    print("\n🔍 检查 langchain_mcp_adapters 模块结构:")
    
    try:
        import langchain_mcp_adapters
        print(f"  模块路径: {langchain_mcp_adapters.__file__}")
        print(f"  模块属性: {dir(langchain_mcp_adapters)}")
        
        import langchain_mcp_adapters.tools as tools_module
        print(f"  tools 模块属性: {dir(tools_module)}")
        
        # 检查 load_mcp_tools 的签名
        import inspect
        sig = inspect.signature(load_mcp_tools)
        print(f"  load_mcp_tools 签名: {sig}")
        
        # 检查文档字符串
        if load_mcp_tools.__doc__:
            print(f"  load_mcp_tools 文档: {load_mcp_tools.__doc__}")
            
    except Exception as e:
        print(f"  ❌ 检查失败: {e}")

# 简化版本：不使用 adapters，直接整合
class SimplifiedMCPService:
    """简化版本，不依赖 langchain_mcp_adapters"""
    
    def __init__(self, env_config: dict, mcp_servers: List[Dict[str, Any]]):
        self.env_config = env_config
        self.mcp_servers = mcp_servers
        self.tools = []
    
    async def load_tools_manually(self):
        """手动加载和包装 MCP 工具"""
        from langchain_core.tools import BaseTool
        from langchain_core.callbacks import CallbackManagerForToolRun
        from pydantic import BaseModel, Field
        
        class SimpleMCPTool(BaseTool):
            """简化的 MCP 工具包装"""
            name: str
            description: str
            mcp_command: str
            mcp_args: List[str]
            
            def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None, **kwargs) -> str:
                # 这里应该调用实际的 MCP 工具
                return f"模拟调用 {self.name} 工具，参数: {kwargs}"
            
            async def _arun(self, **kwargs) -> str:
                return self._run(**kwargs)
        
        # 创建示例工具
        for server_config in self.mcp_servers:
            server_name = server_config.get('name', 'unknown')
            
            # 这里应该连接到实际的 MCP 服务器获取工具列表
            # 现在创建一个示例工具
            example_tool = SimpleMCPTool(
                name=f"{server_name}_example_tool",
                description=f"来自 {server_name} 的示例工具",
                mcp_command=server_config.get('command', ''),
                mcp_args=server_config.get('args', [])
            )
            
            self.tools.append(example_tool)
            print(f"✅ 创建示例工具: {example_tool.name}")
    
    def get_tools(self) -> List[BaseTool]:
        return self.tools

async def main():
    """主调试函数"""
    print("🚀 开始调试 langchain_mcp_adapters...")
    
    # 检查模块结构
    inspect_mcp_adapters()
    
    # 调试 load_mcp_tools
    await debug_load_mcp_tools()
    
    print("\n" + "="*60)
    print("🔄 测试简化版本...")
    
    # 测试简化版本
    test_config = {
        "API_KEY": "test-key",
        "BASE_URL": "https://api.example.com",
        "MODEL": "test-model",
        "MAX_TOKENS": 1000
    }
    
    test_servers = [
        {
            "name": "test-server",
            "command": "echo",
            "args": ["hello"]
        }
    ]
    
    service = SimplifiedMCPService(test_config, test_servers)
    await service.load_tools_manually()
    
    tools = service.get_tools()
    print(f"✅ 创建了 {len(tools)} 个测试工具")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

if __name__ == "__main__":
    asyncio.run(main())