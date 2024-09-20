import os
from ruamel.yaml import YAML


class YamlHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.yaml = YAML()
        self.data = None

    def read(self):
        """读取 YAML 文件并存储数据"""
        if not os.path.exists(self.file_path):
            return FileNotFoundError(f"文件 '{self.file_path}' 不存在.")
        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                self.data = self.yaml.load(file)
            return self.data
        except Exception as e:
            return e

    def write(self):
        """将数据写回 YAML 文件，保留顺序和注释"""
        if self.data is None:
            return ValueError("先读取文件.")
        try:
            with open(self.file_path, "w", encoding="utf-8") as file:
                self.yaml.dump(self.data, file)
        except Exception as e:
            return e

    def get(self, key):
        """根据键获取值"""
        if self.data is None:
            return ValueError("先读取文件.")
        keys = key.split(".")
        value = self.data
        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return KeyError(f"键 '{key}' 不存在.")

    def set(self, key, value):
        """根据键和值修改数据"""
        if self.data is None:
            return ValueError("先读取文件.")
        keys = key.split(".")
        data = self.data
        try:
            for k in keys[:-1]:
                data = data.setdefault(k, {})  # 如果键不存在则创建一个新的字典
            data[keys[-1]] = value
            return self.data
        except KeyError:
            return KeyError(f"键 '{key}' 不存在.")
