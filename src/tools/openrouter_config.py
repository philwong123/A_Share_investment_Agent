import os
import time
import logging
from dotenv import load_dotenv
import requests
import json
from typing import Optional, Dict, Any

# 设置日志记录
logger = logging.getLogger('api_calls')
logger.setLevel(logging.DEBUG)

# 移除所有现有的处理器
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# 创建日志目录
log_dir = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(log_dir, exist_ok=True)

# 设置文件处理器
log_file = os.path.join(log_dir, f'api_calls_{time.strftime("%Y%m%d")}.log')
print(f"Creating log file at: {log_file}")

try:
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
    file_handler.setLevel(logging.DEBUG)
    print("Successfully created file handler")
except Exception as e:
    print(f"Error creating file handler: {str(e)}")

# 设置控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# 设置日志格式
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 添加处理器
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 立即测试日志记录
logger.debug("Logger initialization completed")
logger.info("API logging system started")

# 状态图标
SUCCESS_ICON = "✓"
ERROR_ICON = "✗"
WAIT_ICON = "⟳"

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(project_root, '.env')

# 加载环境变量
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)
    logger.info(f"{SUCCESS_ICON} 已加载环境变量: {env_path}")
else:
    logger.warning(f"{ERROR_ICON} 未找到环境变量文件: {env_path}")

# 定义模型提供商
class AIProvider:
    GEMINI = "gemini"
    ZHIPU = "zhipu"  # 智谱AI

# 配置
DEFAULT_PROVIDER = os.getenv("DEFAULT_AI_PROVIDER", AIProvider.ZHIPU)
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class AIService:
    def __init__(self):
        self.current_provider = DEFAULT_PROVIDER
        
        # 初始化智谱AI
        if ZHIPU_API_KEY:
            self.zhipu_headers = {
                "Authorization": f"Bearer {ZHIPU_API_KEY}",
                "Content-Type": "application/json"
            }
        else:
            logger.warning(f"{ERROR_ICON} 未找到 ZHIPU_API_KEY")
            
        # 初始化Gemini（作为备选）
        if GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel("gemini-1.5-pro")
                logger.info(f"{SUCCESS_ICON} Gemini初始化成功")
            except Exception as e:
                logger.warning(f"{ERROR_ICON} Gemini初始化失败: {str(e)}")
                self.gemini_model = None
    
    def generate_content(self, prompt: str) -> Optional[str]:
        """生成内容，支持故障转移"""
        if self.current_provider == AIProvider.ZHIPU:
            try:
                return self._zhipu_generate(prompt)
            except Exception as e:
                logger.error(f"{ERROR_ICON} 智谱AI调用失败: {str(e)}")
                if self.gemini_model:
                    logger.info(f"{WAIT_ICON} 尝试切换到Gemini...")
                    self.current_provider = AIProvider.GEMINI
                    return self.generate_content(prompt)
                return None
        else:
            try:
                return self._gemini_generate(prompt)
            except Exception as e:
                logger.error(f"{ERROR_ICON} Gemini调用失败: {str(e)}")
                self.current_provider = AIProvider.ZHIPU
                return self.generate_content(prompt)

    def _zhipu_generate(self, prompt: str) -> Optional[str]:
        """调用智谱AI API"""
        url = "https://open.bigmodel.cn/api/paas/v3/model-api/chatglm_turbo/invoke"
        
        payload = {
            "prompt": prompt,
            "temperature": 0.7,
            "top_p": 0.7,
            "request_id": f"request_{int(time.time())}"
        }
        
        try:
            response = requests.post(url, headers=self.zhipu_headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if "data" in result and "choices" in result["data"]:
                return result["data"]["choices"][0]["content"]
            else:
                logger.error(f"{ERROR_ICON} 智谱AI返回异常响应: {result}")
                return None
                
        except Exception as e:
            logger.error(f"{ERROR_ICON} 智谱AI API调用失败: {str(e)}")
            raise

    def _gemini_generate(self, prompt: str) -> Optional[str]:
        """调用Gemini API"""
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text if response and hasattr(response, 'text') else None
        except Exception as e:
            logger.error(f"{ERROR_ICON} Gemini API调用失败: {str(e)}")
            raise

# 创建全局AI服务实例
ai_service = AIService()

def get_chat_completion(messages, model=None, max_retries=3, initial_retry_delay=1):
    """获取聊天完成结果"""
    try:
        # 转换消息格式
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"

        # 调用AI服务
        for attempt in range(max_retries):
            try:
                response = ai_service.generate_content(prompt.strip())
                if response:
                    return response
                
                if attempt < max_retries - 1:
                    retry_delay = initial_retry_delay * (2 ** attempt)
                    logger.info(f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    
            except Exception as e:
                logger.error(f"{ERROR_ICON} 尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
                if attempt < max_retries - 1:
                    retry_delay = initial_retry_delay * (2 ** attempt)
                    logger.info(f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)

        return None

    except Exception as e:
        logger.error(f"{ERROR_ICON} get_chat_completion 发生错误: {str(e)}")
        return None
