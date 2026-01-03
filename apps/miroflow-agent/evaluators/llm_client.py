"""
LLM 客户端 - 封装 OpenAI API 调用
"""

import json
from typing import Dict, Optional
from openai import OpenAI
from .base import EvalConfig


class LLMClient:
    """LLM 客户端"""
    
    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
        self.client = OpenAI(
            api_key=self.config.api_key, 
            base_url=self.config.base_url
        )
        self.call_count = 0
    
    def call(self, system: str, user: str, json_mode: bool = True) -> Optional[Dict]:
        """调用 LLM
        
        Args:
            system: 系统提示
            user: 用户消息
            json_mode: 是否要求 JSON 格式输出
            
        Returns:
            解析后的响应字典，失败返回 None
        """
        self.call_count += 1
        try:
            kwargs = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                "temperature": self.config.temperature
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            
            return json.loads(content) if json_mode else {"text": content}
        except Exception as e:
            print(f"  ⚠️ LLM 调用错误: {e}")
            return None
    
    def call_with_retry(self, system: str, user: str, 
                        json_mode: bool = True, max_retries: int = 3) -> Optional[Dict]:
        """带重试的 LLM 调用"""
        for attempt in range(max_retries):
            result = self.call(system, user, json_mode)
            if result is not None:
                return result
            print(f"  重试 {attempt + 1}/{max_retries}...")
        return None
