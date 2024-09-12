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

    def update(self, key, value):
        """更新数据中的指定键值对"""
        if self.data is None:
            return ValueError("先读取文件.")
        try:
            self.data[key] = value
        except Exception as e:
            return e

    def get(self, key):
        """获取数据中的指定键的值"""
        if self.data is None:
            return ValueError("先读取文件.")
        try:
            return self.data.get(key)
        except Exception as e:
            return e
