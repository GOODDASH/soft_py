import re
import serial
import threading
import time
from logging import info, warning, error


class SerialPortReader:
    def __init__(self, port, baud_rate=9600):
        self.port = port
        self.baud_rate = baud_rate
        try:
            info("尝试打开串口")
            self.ser = serial.Serial(port, baud_rate, timeout=0)
            info("串口打开成功")
        except serial.SerialException as e:
            error(f"无法打开串口: {e}")
            raise e

        self.latest_data = 0.0  # 初始值设为0.0，假设数据是浮点数
        self.running = True
        self.thread = threading.Thread(target=self._listen)
        self.thread.start()

    def _listen(self):
        """后台线程，持续监听串口数据并尝试转换为float"""
        try:
            while self.running:
                if self.ser.in_waiting:
                    data_str = self.ser.read(self.ser.in_waiting).decode("utf-8").strip()
                    # print(data_str)  # "1 MW +011.0105 mm"
                    match = re.search(r"([+|-]\d+\.\d+)", data_str)
                    try:
                        self.latest_data = float(match.group())
                    except ValueError:
                        # 如果转换失败（例如，接收到的数据不是合法的浮点数字符串），打印错误并忽略此次数据
                        warning(f"无法将接收到的数据转换为float: '{data_str}'")
                time.sleep(0.1)
        except Exception as e:
            error(f"监听发生错误: {e}")
        finally:
            if self.ser:
                self.ser.close()

    def read(self):
        """返回最近一次成功转换的浮点数"""
        return self.latest_data

    def stop(self):
        """停止监听并关闭串口"""
        self.running = False
        self.thread.join()


# .\.env\Scripts\python.exe -m src.core.read_error
if __name__ == "__main__":
    try:
        port_reader = SerialPortReader("COM3")  # 根据实际情况替换'COM1'和波特率
        try:
            while True:
                data = port_reader.read()
                print(f"收到数据:{data:.4f}")
                time.sleep(1)
        finally:
            port_reader.stop()
    except serial.SerialException as e:
        print(f"无法打开串口: {e}")
