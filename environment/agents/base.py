import json

from pydantic import BaseModel, ConfigDict
from typing import Type, Dict, get_origin
from collections import OrderedDict

def _get_type_name(annotation: Type) -> str:
    """处理复杂类型名称提取"""
    if origin := get_origin(annotation):
        return origin.__name__
    return annotation.__name__


# function_registry.py
import json
import inspect
import importlib
from pathlib import Path
from collections import OrderedDict
from typing import Type, Dict, List


class FunctionRegistry:
    """
    增强版函数注册器，支持自动发现工具类
    """
    _registry = OrderedDict()
    _initialized = False

    @classmethod
    def register(cls, tool_cls):
        """注册单个工具类"""
        tool_info = {
            "name": tool_cls.__name__,
            "description": (tool_cls.__doc__ or "").strip(),
            "input_params": tool_cls.get_input_schema(),
            "output_params": tool_cls.get_output_schema()
        }
        cls._registry[tool_cls.__name__] = tool_info
        return tool_cls

    @classmethod
    def auto_register(cls, package_dir: str = "environment/roles"):
        """
        自动注册指定目录下的所有工具类
        :param package_dir: 工具类所在的包路径
        """
        if cls._initialized:
            return

        tools_dir = Path(package_dir)
        if not tools_dir.exists():
            # Resolve against the project root so VideoAgent runs from any cwd.
            from environment.config.runtime import PROJECT_ROOT
            tools_dir = PROJECT_ROOT / package_dir
        if not tools_dir.exists():
            raise ValueError(f"Tools directory not found: {tools_dir}")

        # Make sure tools_dir is absolute
        tools_dir = tools_dir.resolve()

        # A lean install has no TTS/SVC stack, so those roles cannot import.
        # That is expected: collect them and print one summary instead of a
        # traceback per file.
        skipped = {}

        from environment.config.runtime import VOICE_ROLES, voice_stack_available
        voice_ready = voice_stack_available()

        # 动态导入所有模块
        for module_file in tools_dir.glob("**/*.py"):
            if module_file.name.startswith("_"):
                continue

            relative_path = module_file.relative_to(tools_dir)
            module_path = str(relative_path).replace("\\", "/").replace(".py", "").replace("/", ".")
            base_package = package_dir.replace("/", ".")
            module_name = f"{base_package}.{module_path}"

            try:
                module = importlib.import_module(module_name)
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and
                            issubclass(obj, BaseTool) and
                            obj != BaseTool):
                        # These import cleanly but cannot execute without the
                        # voice models; hide them so the router never plans a
                        # workflow that dies partway through.
                        if not voice_ready and obj.__name__ in VOICE_ROLES:
                            skipped.setdefault("voice models", []).append(obj.__name__)
                            continue
                        cls.register(obj)
            except ImportError as e:
                missing = getattr(e, "name", None) or str(e)
                skipped.setdefault(missing, []).append(module_path)
            except Exception as e:
                skipped.setdefault(f"{type(e).__name__}: {e}", []).append(module_path)

        if skipped:
            total = sum(len(v) for v in skipped.values())
            print(f"[roles] {len(cls._registry)} tools registered, {total} unavailable:")
            for reason, modules in sorted(skipped.items()):
                shown = ", ".join(sorted(modules)[:4])
                more = f" (+{len(modules) - 4} more)" if len(modules) > 4 else ""
                print(f"        missing '{reason}' -> {shown}{more}")
            print("        This is expected on a lean install; see SETUP.md.")

        cls._initialized = True

    @classmethod
    def get_registry(cls):
        """
        获取注册表
        :输出格式 ('json')
        """
        if not cls._initialized:
            cls.auto_register()

        sorted_registry = OrderedDict(sorted(cls._registry.items()))
        return json.dumps(sorted_registry, indent=2, ensure_ascii=False)

class BaseTool:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    class BaseInputSchema(BaseModel):
        model_config = ConfigDict(extra='ignore')

    InputSchema: Type[BaseModel] = BaseInputSchema
    OutputSchema: Type[BaseModel]

    @classmethod
    def get_input_schema(cls) -> Dict[str, Dict]:
        """生成输入参数描述字典"""
        return {
            field: {
                "type": _get_type_name(field_info.annotation),
                "description": field_info.description
            }
            for field, field_info in cls.InputSchema.model_fields.items()
        }

    @classmethod
    def get_output_schema(cls) -> Dict[str, Dict]:
        """生成输出参数描述字典"""
        return {
            field: {
                "type": _get_type_name(field_info.annotation),
                "description": field_info.description
            }
            for field, field_info in cls.OutputSchema.model_fields.items()
        }


    def execute(self, **kwargs):
        raise NotImplementedError