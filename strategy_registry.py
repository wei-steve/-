import importlib
import inspect
import pkgutil
import logging
from typing import Dict, Type
from strategy_interface import IStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StrategyRegistry:
    def __init__(self):
        self._strategies: Dict[str, Type[IStrategy]] = {}

    def register_strategy(self, strategy_name: str, strategy_class: Type[IStrategy]) -> None:
        if not issubclass(strategy_class, IStrategy):
            raise ValueError(f"策略类 {strategy_class.__name__} 必须实现 IStrategy 接口")
        self._strategies[strategy_name] = strategy_class
        logger.info(f"注册策略: {strategy_name}")

    def get_strategy(self, strategy_name: str) -> Type[IStrategy]:
        strategy_class = self._strategies.get(strategy_name)
        if strategy_class is None:
            raise KeyError(f"未找到策略: {strategy_name}")
        return strategy_class

    def get_all_strategies(self) -> Dict[str, Type[IStrategy]]:
        return self._strategies

    def discover_strategies(self, package_name: str) -> None:
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

# 模块级别的控制逻辑
_REGISTRY_INITIALIZED = False
registry = StrategyRegistry()

if not _REGISTRY_INITIALIZED:
    registry.discover_strategies("strategies")
    _REGISTRY_INITIALIZED = True