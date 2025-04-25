from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Tuple, Any

class IStrategy(ABC):
    @property
    @abstractmethod
    def display_name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def params(self) -> Dict[str, float]:
        pass

    @property
    @abstractmethod
    def param_schema(self) -> Dict[str, Dict[str, Any]]:
        pass

    @abstractmethod
    def compute_indicator(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        pass

    @abstractmethod
    def indicator_config(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
        pass