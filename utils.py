import logging

logger = logging.getLogger(__name__)

def serialize_content(content: any) -> str:
    """将任意内容序列化为 JSON 可序列化的格式"""
    logger.debug(f"开始序列化内容，类型: {type(content).__module__}.{type(content).__name__}")
    try:
        if isinstance(content, (str, dict, list, int, float, bool, type(None))):
            logger.debug("内容为基本类型，直接返回")
            return content
        elif hasattr(content, 'text'):
            logger.debug("检测到 text 属性，返回 content.text")
            return content.text
        elif hasattr(content, 'value'):
            logger.debug("检测到 value 属性，返回 content.value")
            return content.value
        elif hasattr(content, 'model_dump'):
            logger.debug("检测到 model_dump 方法，返回 content.model_dump()")
            return content.model_dump()
        elif hasattr(content, 'dict'):
            logger.debug("检测到 dict 方法，返回 content.dict()")
            return content.dict()
        elif hasattr(content, '__dict__'):
            logger.debug("检测到 __dict__ 属性，返回 content.__dict__")
            return content.__dict__
        else:
            logger.warning(f"未知类型 {type(content).__module__}.{type(content).__name__}，尝试转换为字符串")
            return str(content)
    except Exception as e:
        logger.error(f"序列化内容失败: {str(e)}")
        return str(content)