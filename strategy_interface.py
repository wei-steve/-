from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Tuple, Any

class IStrategy(ABC):
    @property
    @abstractmethod
    def display_name(self) -> str:
        """返回策略的显示名称"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """返回策略的描述"""
        pass

    @property
    @abstractmethod
    def params(self) -> Dict[str, float]:
        """返回策略的默认参数"""
        pass

    @property
    @abstractmethod
    def param_schema(self) -> Dict[str, Dict[str, Any]]:
        """返回策略参数的模式（用于输入验证和描述）"""
        pass

    @abstractmethod
    def compute_indicator(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """计算策略的指标值"""
        pass

    @abstractmethod
    def indicator_config(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """返回指标的配置（如水平线）"""
        pass

    @abstractmethod
    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
        """运行回测，返回统计数据、买入信号和卖出信号"""
        pass