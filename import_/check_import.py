from importlib.util import find_spec
from importlib.metadata import version
def isImport(modules:list[str])->None:
    """
    输入库列表，检查是否安装
    """
    for module in modules:
        if find_spec(module) is None:
            raise Exception(f"{module}未安装")
        else:
            print(f"{module}版本：",version(module))
