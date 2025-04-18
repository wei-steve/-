import importlib
import inspect
import pkgutil
import logging
from typing import Dict, Type
from strategy_interface import IStrategy

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StrategyRegistry:
    def __init__(self):
        self._strategies: Dict[str, Type[IStrategy]] = {}

    def register_strategy(self, strategy_name: str, strategy_class: Type[IStrategy]) -> None:
        """注册一个策略类"""
        if not issubclass(strategy_class, IStrategy):
            raise ValueError(f"策略类 {strategy_class.__name__} 必须实现 IStrategy 接口")
        self._strategies[strategy_name] = strategy_class
        logger.info(f"注册策略: {strategy_name}")

    def get_strategy(self, strategy_name: str) -> Type[IStrategy]:
        """根据名称获取策略类"""
        strategy_class = self._strategies.get(strategy_name)
        if strategy_class is None:
            raise KeyError(f"未找到策略: {strategy_name}")
        return strategy_class

    def get_all_strategies(self) -> Dict[str, Type[IStrategy]]:
        """获取所有已注册的策略"""
        return self._strategies

    def discover_strategies(self, package_name: str) -> None:
        """自动发现并注册指定包中的所有策略"""
        try:
            package = importlib.import_module(package_name)
            for _, module_name, _ in pkgutil.iter_modules(package.__path__):
                try:
                    module = importlib.import_module(f"{package_name}.{module_name}")
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and issubclass(obj, IStrategy) and obj != IStrategy:
                            self.register_strategy(module_name, obj)
                except Exception as e:
                    logger.error(f"加载模块 {module_name} 失败: {e}")
        except Exception as e:
            logger.error(f"发现策略失败，包 {package_name}: {e}")

# 创建全局 registry 实例
registry = StrategyRegistry()

# 自动发现 strategies 包中的所有策略
registry.discover_strategies("strategies")