import time
from logging import info, error
from pymodbus.client import ModbusTcpClient


class ModbusTCP:
    def __init__(self, paras):
        # paras: [(ip, slave, port, addr, reg_num, exclude_idx), ...]
        self.paras = paras
        self.clients = []

    def connect_all(self):
        for para in self.paras:
            try:
                client = ModbusTcpClient(host=para[0], port=para[2])
                if client.connect():
                    self.clients.append(client)
                    info(f"连接{para[0]}: {para[2]}成功")
                else:
                    error(f"连接{para[0]}: {para[2]}失败")
                    return (False, f"连接{para[0]}: {para[2]}失败")
            except Exception as e:
                error(f"连接{para[0]}: {para[2]}失败: {e}")
                return (False, f"连接{para[0]}: {para[2]}失败")
        return (True, f"连接成功")

    def close_all(self):
        for client in self.clients:
            client.close()
        self.clients.clear()

    def read_temperature(self) -> list:
        all_registers = []
        for index, client in enumerate(self.clients):
            try:
                response = client.read_holding_registers(
                    address=self.paras[index][3],
                    count=self.paras[index][4],
                    slave=self.paras[index][1],
                )
                if response.isError():
                    error(f"读取寄存器失败: {response}")
                    all_registers = []  # 读取失败置为空列表
                else:
                    registers = response.registers

                    exlude_idx_str = self.paras[index][5]
                    if exlude_idx_str:
                        exclude_idx = list(map(int, self.paras[index][5].split(",")))
                        registers = [
                            val for index, val in enumerate(registers) if index not in exclude_idx
                        ]
                    all_registers.extend(self._trans_func(registers))
            except Exception as e:
                error(f"读取寄存器发生异常: {e}")
        return all_registers

    def _trans_func(self, registers):
        return [round(134 * val / 4095 - 50, 2) for val in registers]


# .\.env\Scripts\python.exe -m src.core.modbus_tcp
if __name__ == "__main__":
    paras = [("127.0.0.1", 255, 502, 620, 5)]
    client = ModbusTCP(paras)
    if client.connect_all():
        try:
            while True:
                all_registers = client.read_temperature()
                print("读取的所有温度值:", all_registers)
                time.sleep(2)
        except KeyboardInterrupt:
            pass
        finally:
            client.close_all()
    else:
        print("连接所有设备失败.")
