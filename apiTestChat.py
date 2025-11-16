import asyncio
import aiohttp
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(f"./logs/test_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8110"

async def test_chat_endpoint(message: str):
    """Test the /chat endpoint with streaming response."""
    url = f"{API_BASE_URL}/chat"
    payload = {
        "message": message,
        "use_mcp": True  # 保持兼容性，虽然这个参数已废弃
    }

    async with aiohttp.ClientSession() as session:
        try:
            logger.info(f"Sending chat request: {message[:50]}...")
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Chat request failed with status {response.status}: {error_text}")
                    print(f"❌ Error: {response.status} - {error_text}")
                    return

                print("\n AI response:")
                print("-" * 50)
                async for line in response.content:
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            chunk_type = chunk.get("type")
                            content = chunk.get("content")
                            print(content)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse chunk: {line.decode('utf-8')}: {e}")
                            print(f"\n❌ Failed to parse chunk: {line.decode('utf-8')}")
                print("-" * 50)

        except aiohttp.ClientError as e:
            logger.error(f"HTTP request failed: {e}")
            print(f"❌ HTTP Error: {e}")

async def test_chat_debug_endpoint(message: str, session_id: str = "debug_session") -> None:
    """Test the /chat/debug endpoint with non-streaming response."""
    url = f"{API_BASE_URL}/chat/debug"
    payload = {
        "message": message,
        "session_id": session_id
    }

    async with aiohttp.ClientSession() as session:
        try:
            logger.info(f"Sending debug chat request: {message[:50]}...")
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Debug chat request failed with status {response.status}: {error_text}")
                    print(f"❌ Error: {response.status} - {error_text}")
                    return

                result = await response.json()
                print("\n🔍 Debug Response:")
                print(f"Success: {result.get('success')}")
                print(f"Session ID: {result.get('session_id')}")
                print(f"MCP Enabled: {result.get('mcp_enabled')}")
                print(f"Available Functions Count: {result.get('available_functions_count')}")
                print(f"Implementation: {result.get('implementation')}")
                print(f"Response Count: {result.get('response_count')}")
                
                if result.get('mcp_functions'):
                    print(f"MCP Functions: {', '.join(result.get('mcp_functions', []))}")
                
                if result.get('responses'):
                    print("Responses:")
                    for i, resp in enumerate(result.get('responses', []), 1):
                        print(f"  {i}. {resp.get('type')}: {resp.get('content', '')[:100]}...")
                
                if result.get('error'):
                    print(f"Error: {result.get('error')}")
                print("-" * 50)

        except aiohttp.ClientError as e:
            logger.error(f"HTTP request failed: {e}")
            print(f"❌ HTTP Error: {e}")

async def test_status_endpoint() -> None:
    """Test the /test endpoint to check API status."""
    url = f"{API_BASE_URL}/test"
    async with aiohttp.ClientSession() as session:
        try:
            logger.info("Testing /test endpoint...")
            async with session.get(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Status request failed with status {response.status}: {error_text}")
                    print(f"❌ Error: {response.status} - {error_text}")
                    return

                result = await response.json()
                print("\n📊 API Status:")
                print(f"Message: {result.get('message')}")
                print(f"Timestamp: {result.get('timestamp')}")
                print(f"MCP Enabled: {result.get('mcp_enabled')}")
                print(f"MCP Functions Count: {result.get('mcp_functions_count')}")
                print(f"Service Status: {result.get('service_status')}")
                print(f"Implementation: {result.get('implementation')}")
                if 'error' in result:
                    print(f"Error: {result.get('error')}")
                print("-" * 50)

        except aiohttp.ClientError as e:
            logger.error(f"HTTP request failed: {e}")
            print(f"❌ HTTP Error: {e}")

async def test_mcp_status_endpoint() -> None:
    """Test the /mcp/status endpoint to check MCP server status."""
    url = f"{API_BASE_URL}/mcp/status"
    async with aiohttp.ClientSession() as session:
        try:
            logger.info("Testing /mcp/status endpoint...")
            async with session.get(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"MCP status request failed with status {response.status}: {error_text}")
                    print(f"❌ Error: {response.status} - {error_text}")
                    return

                result = await response.json()
                print("\n🛠️ MCP Status:")
                print(f"MCP Enabled: {result.get('mcp_enabled')}")
                print(f"MCP Servers Count: {result.get('mcp_servers_count')}")
                print(f"Working Servers: {result.get('working_servers')}/{result.get('total_servers')}")
                print(f"Clients Initialized: {result.get('clients_initialized')}")
                print(f"Implementation: {result.get('implementation')}")
                print(f"Timestamp: {result.get('timestamp')}")
                
                # 显示功能类型统计
                if result.get('function_types'):
                    print("Function Types:")
                    for func_type, count in result.get('function_types', {}).items():
                        print(f"  - {func_type.capitalize()}: {count}")
                
                print("Functions:")
                for func in result.get('mcp_functions', []):
                    func_type = func.get('type', 'unknown').upper()
                    print(f"  - [{func_type}] {func['name']}: {func['description']}")
                    if func_type == 'RESOURCE' and func.get('uri'):
                        print(f"    URI: {func['uri']}")
                
                print("Connection Status:")
                for server, status in result.get('connection_status', {}).items():
                    print(f"  - {server}: {'✅ Connected' if status else '❌ Disconnected'}")
                print("-" * 50)

        except aiohttp.ClientError as e:
            logger.error(f"HTTP request failed: {e}")
            print(f"❌ HTTP Error: {e}")

async def test_mcp_functions_endpoint() -> None:
    """Test the /mcp/functions endpoint to list available functions."""
    url = f"{API_BASE_URL}/mcp/functions"
    async with aiohttp.ClientSession() as session:
        try:
            logger.info("Testing /mcp/functions endpoint...")
            async with session.get(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"MCP functions request failed with status {response.status}: {error_text}")
                    print(f"❌ Error: {response.status} - {error_text}")
                    return

                result = await response.json()
                print("\n🔧 Available MCP Functions:")
                print(f"Total Count: {result.get('total_count')}")
                print(f"Implementation: {result.get('implementation')}")
                print(f"Function Types: {', '.join(result.get('function_types', []))}")
                print(f"Timestamp: {result.get('timestamp')}")
                
                # 按类型分组显示
                if result.get('grouped_functions'):
                    print("\nFunctions by Type:")
                    for func_type, functions in result.get('grouped_functions', {}).items():
                        print(f"\n  {func_type.upper()}:")
                        for func in functions:
                            print(f"    - {func['name']}: {func['description']}")
                            if func_type == 'resource' and func.get('uri'):
                                print(f"      URI: {func['uri']}")
                            elif func_type == 'prompt' and func.get('arguments'):
                                args = [arg['name'] for arg in func.get('arguments', [])]
                                if args:
                                    print(f"      Args: {', '.join(args)}")
                else:
                    print("Functions:")
                    for func in result.get('functions', []):
                        func_type = func.get('type', 'unknown').upper()
                        print(f"  - [{func_type}] {func['name']}: {func['description']}")
                print("-" * 50)

        except aiohttp.ClientError as e:
            logger.error(f"HTTP request failed: {e}")
            print(f"❌ HTTP Error: {e}")

async def test_mcp_reload_endpoint() -> None:
    """Test the /mcp/reload endpoint to reload MCP configuration."""
    url = f"{API_BASE_URL}/mcp/reload"
    async with aiohttp.ClientSession() as session:
        try:
            logger.info("Testing /mcp/reload endpoint...")
            async with session.post(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"MCP reload request failed with status {response.status}: {error_text}")
                    print(f"❌ Error: {response.status} - {error_text}")
                    return

                result = await response.json()
                print("\n🔄 MCP Reload Results:")
                print(f"Success: {result.get('success')}")
                print(f"Message: {result.get('message')}")
                print(f"Config Path: {result.get('config_path')}")
                print(f"Loaded Servers: {result.get('loaded_servers')}")
                print(f"Working Servers: {result.get('working_servers')}")
                print(f"Total Functions: {result.get('total_functions')}")
                print(f"Implementation: {result.get('implementation')}")
                
                print("Connection Results:")
                for server, status in result.get('connection_results', {}).items():
                    print(f"  - {server}: {'✅ Connected' if status else '❌ Disconnected'}")
                print("-" * 50)

        except aiohttp.ClientError as e:
            logger.error(f"HTTP request failed: {e}")
            print(f"❌ HTTP Error: {e}")

async def interactive_test():
    """Run an interactive test loop for the API."""
    print("\n🤖 API Testing Client - Stream-based MCP Implementation")
    print("Available commands:")
    print("  - Enter a message to test /chat endpoint")
    print("  - 'debug <message>' to test /chat/debug endpoint")
    print("  - 'status' to test /test endpoint")
    print("  - 'mcp' to test /mcp/status endpoint")
    print("  - 'functions' to test /mcp/functions endpoint")
    print("  - 'reload' to test /mcp/reload endpoint")
    print("  - 'quit' or 'exit' to exit")
    print("-" * 50)

    session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    while True:
        try:
            user_input = input("\n💬 Enter command or message: ").strip()
            print("this is what i enter: "+user_input)
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Exiting...")
                break
            elif user_input.lower() == 'status':
                await test_status_endpoint()
            elif user_input.lower() == 'mcp':
                await test_mcp_status_endpoint()
            elif user_input.lower() in ['functions', 'tools']:
                await test_mcp_functions_endpoint()
            elif user_input.lower() == 'reload':
                await test_mcp_reload_endpoint()
            elif user_input.lower().startswith('debug '):
                debug_message = user_input[6:].strip()
                if debug_message:
                    await test_chat_debug_endpoint(debug_message, session_id)
                else:
                    print("⚠️ Please provide a message after 'debug'")
            elif user_input:
                await test_chat_endpoint(user_input)
            else:
                print("⚠️ Please enter a valid command or message.")
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user, exiting...")
            break
        except Exception as e:
            logger.error(f"Error in interactive test: {e}")
            print(f"❌ Error: {e}")

async def run_batch_tests():
    """Run a batch of predefined tests."""
    print("\n🧪 Running Batch Tests")
    print("=" * 50)
    
    # Test API status
    print("\n1. Testing API Status...")
    await test_status_endpoint()
    
    # Test MCP status
    print("\n2. Testing MCP Status...")
    await test_mcp_status_endpoint()
    
    # Test MCP functions list
    print("\n3. Testing MCP Functions List...")
    await test_mcp_functions_endpoint()
    
    # Test simple chat
    print("\n4. Testing Simple Chat...")
    await test_chat_endpoint("Hello, how are you?")
    
    # Test MCP-related chat
    print("\n5. Testing MCP Function Call Chat...")
    await test_chat_endpoint("Can you help me with file operations?")
    
    # Test debug endpoint
    print("\n6. Testing Debug Endpoint...")
    await test_chat_debug_endpoint("What tools are available?")
    
    print("\n✅ Batch tests completed!")
    print("=" * 50)

if __name__ == "__main__":
    try:
        print("Choose test mode:")
        print("1. Interactive mode")
        print("2. Batch tests")
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "2":
            asyncio.run(run_batch_tests())
        else:
            asyncio.run(interactive_test())
            
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Program failed: {e}", exc_info=True)