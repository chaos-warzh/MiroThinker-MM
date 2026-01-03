"""
基础类和配置 - 所有评估器共享的基础设施
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


@dataclass
class EvalConfig:
    """评估配置 - 使用阿里云 API"""
    api_key: str = field(default_factory=lambda: os.getenv(
        "ALIBABA_API_KEY", os.getenv("OPENAI_API_KEY", "")
    ))
    base_url: str = field(default_factory=lambda: os.getenv(
        "ALIBABA_BASE_URL", os.getenv("OPENAI_BASE_URL", "")
    ))
    model_name: str = field(default_factory=lambda: os.getenv(
        "ALIBABA_MODEL", "gpt-51-1113-global"
    ))
    max_segment_length: int = 8000
    temperature: float = 0.1


@dataclass
class EvalResult:
    """评估结果"""
    metric_name: str
    score: float  # 0-100
    details: Dict[str, Any]
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric_name': self.metric_name,
            'score': self.score,
            'weight': self.weight,
            'details': self.details
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class BaseEvaluator(ABC):
    """评估器基类"""
    
    metric_name: str = "base"
    weight: float = 1.0
    
    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
    
    @abstractmethod
    def evaluate(self, result_text: str, **kwargs) -> EvalResult:
        """执行评估，返回评估结果"""
        pass
    
    def load_json(self, path: Path) -> Any:
        """加载 JSON 文件"""
        return json.loads(path.read_text(encoding='utf-8'))
    
    def save_result(self, result: EvalResult, output_path: Path) -> None:
        """保存评估结果"""
        output_path.write_text(result.to_json(), encoding='utf-8')
