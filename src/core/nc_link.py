import uuid
import threading
import json
import paho.mqtt.client as mqtt_client  # version==1.6.1
from logging import info, warning, error


class NCLink:
    def __init__(self, mqtt_ip: str, mqtt_port: int, sn: str, client_id=None) -> None:
        self.mqtt_ip = mqtt_ip
        self.mqtt_port = mqtt_port
        self.sn = sn
        if not client_id:
            self.client_id = str(uuid.uuid1())
            info(f"没有提供client_id, 已自动创建id{self.client_id}")
        else:
            self.client_id = client_id
        self.request_number = 0
        self.request_locks = dict()
        self.lock_dict_operation_lock = threading.Lock()
        self.topics = {
            "Set/Response/+/" + self.client_id,
            "Query/Response/+/" + self.client_id,
        }
        self.id_name_hash = self.read_json()

    class request_lock:
        def __init__(self, id: str) -> None:
            self.id = id
            self.lock = threading.Lock()
            self.cond = threading.Condition(self.lock)
            self.response = None

    def request_id(self):
        self.request_number = self.request_number + 1
        return self.client_id + str(self.request_number)

    @staticmethod
    def on_mqtt_connect(client: mqtt_client.Client, nclinkObj, flags, rc):
        for t in nclinkObj.topics:
            client.subscribe((t, 0))
        return

    @staticmethod
    def on_mqtt_disconnect(client, nclinkObj, rc):
        return

    @staticmethod
    def on_mqtt_message(client, nclinkObj, message: mqtt_client.MQTTMessage):
        if message.topic.startswith("Set/Response/") or message.topic.startswith("Query/Response/"):
            resp_obj = json.loads(str(message.payload, "utf-8"))
            if (
                isinstance(resp_obj, dict)
                and "@id" in resp_obj
                and isinstance(resp_obj["@id"], str)
            ):
                id = resp_obj["@id"]  # 找到返回消息对应的id
                nclinkObj.lock_dict_operation_lock.acquire()
                if id in nclinkObj.request_locks:  # 如果id在请求锁里面
                    lock = nclinkObj.request_locks[id]
                    lock.lock.acquire()
                    nclinkObj.lock_dict_operation_lock.release()
                    lock.response = resp_obj
                    lock.cond.notify()  # 通知之前的锁
                    lock.lock.release()
                    return
                nclinkObj.lock_dict_operation_lock.release()

        return

    @staticmethod
    def read_json():
        id_name_hash = {}

        try:
            with open("./res/model.json", "r", encoding="gbk", errors="ignore") as file:
                info("打开model.json文件成功")
                json_data = json.load(file)

                # 读取JSON中的数据项并填充哈希表
                if "devices" in json_data and isinstance(json_data["devices"], list):
                    components = json_data["devices"][0]["components"]
                    if isinstance(components, list) and len(components) >= 5:
                        data_items = components[4].get("dataItems", [])
                        for item in data_items:
                            if isinstance(item, dict) and "number" in item and "id" in item:
                                name = str(item["number"])
                                id_str = str(item["id"])
                                id_name_hash[name] = id_str

                        info("解析model.json成功")
                    else:
                        warning("components不是列表或长度不足")
                else:
                    warning("devices不存在或不是列表")

        except FileNotFoundError:
            error("文件不存在")
        except json.JSONDecodeError:
            error("JSON解码错误")

        return id_name_hash

    def send_request(
        self,
        client: mqtt_client.Client,
        topic: str,
        payload: dict,
        timeoutMs: int = 1000,
    ):
        id = self.request_id()  # 获取每个请求唯一的ID
        lock = self.request_lock(id)  # 根据ID创建请求锁
        self.lock_dict_operation_lock.acquire()  # 获取 (查找请求锁) 的锁
        self.request_locks[id] = lock  # 将这个锁放进对应ID的字典
        lock.lock.acquire()  # 获取请求锁
        self.lock_dict_operation_lock.release()  # 已经获取请求锁, 释放 (查找请求锁) 的锁
        payload["@id"] = id  # 给负载加上id标识
        client.publish(topic, json.dumps(payload))  # 解析负载并发布消息
        info(f"已发布topic: topic, payload: {payload} 的请求消息")
        lock.cond.wait(timeoutMs / 1000.0)
        # 如果收到消息, 就会调用on_mqtt_message, 根据消息中的id消息通知对应的锁释放
        self.lock_dict_operation_lock.acquire()
        lock.lock.release()  # 如果条件锁一直没释放(没收到消息), 也会自动释放锁
        response = None
        state = False
        if lock.response:  # 如果收到消息
            state = True
            response = lock.response
            info(f"成功收到响应消息{response}")
        self.request_locks.pop(id)  # 将这个id的锁删掉
        self.lock_dict_operation_lock.release()
        return response, state

    def connect(self):
        self.client = mqtt_client.Client(self.client_id, userdata=self)
        # 回调函数, 注意理解回调函数
        self.client.on_connect = self.on_mqtt_connect
        self.client.on_disconnect = self.on_mqtt_disconnect
        self.client.on_message = self.on_mqtt_message
        result_code = None
        try:
            result_code = self.client.connect(self.mqtt_ip, self.mqtt_port, 60)
            if result_code == 0:
                info("连接MQTT服务器成功")
                self.client.loop_start()
                return (True, "连接MQTT服务器成功")
            else:
                error("连接MQTT服务器失败")
                return (False, "连接MQTT服务器失败")
        except Exception as e:
            error("连接MQTT服务器发生异常: {}".format(e))
            return (False, f"连接MQTT服务器发生异常: {e}")

    def disconnect(self):
        self.client.disconnect()
        self.client.loop_stop()

    def parse_query_requests(self, name_indexes):
        # 解析每个输入项，分离id和indexes
        values_items = []
        for item in name_indexes:
            parts = item.split(":")
            if len(parts) != 2:
                return f'输入格式有错: {item} , 格式为应为["REG_G:0-2"]'

            id_ = self.id_name_hash.get(parts[0])
            if id_ is None:
                return "未找到对应id, 请检查"

            indexes = parts[1].split(",")

            values_items.append({"id": id_, "params": {"indexes": indexes}})

        # 构建最终的JSON对象
        result = {"ids": values_items}

        # info(f"解析取值请求消息成功: {result}")
        return result

    def parse_set_requests(self, name_indexes):
        # 解析每个输入项，分离id, index和value
        values_items = []
        for item in name_indexes:
            parts = item.split(":")
            if len(parts) != 3:
                return f'输入格式有错: {item} , 格式为应为["REG_G:3080:20"]'

            id_ = self.id_name_hash.get(parts[0])
            if id_ is None:
                return "未找到对应id, 请检查"
            
            try:
                index = int(parts[1])
            except ValueError:
                return "index输入了不能转换为整数的字符"

            try:
                value = int(parts[2])
            except ValueError:
                return "value输入了不能转换为整数的字符"

            values_items.append({"id": id_, "params": {"index": index, "value": value}})

        # 构建最终的JSON对象
        result = {"values": values_items}

        # info("解析设值请求消息成功:", json.dumps(result))
        return result

    def set(self, items, need_parse=True):
        if need_parse == True:
            req_items = self.parse_set_requests(items)
            print(req_items)
        else:
            req_items = items
        data, success = self.send_request(
            self.client, "Set/Request/" + self.sn + "/" + self.client_id, req_items
        )
        if not success:
            return None
        results = data["results"]
        # FIXME: 搞清楚len(values)和len(items)关系
        # if not results or not isinstance(results, list) or len(results) != len(items):
        #     return None

        states = []

        for e in results:
            if not isinstance(e, dict):
                states.append(False)
                continue

            ss = e.get("code", None)
            if ss == "OK":
                states.append(True)
            elif ss == "NG":
                states.append(False)
            else:
                states.append(None)

        return states

    def query(self, items, timeout=200, need_parse=True):
        if need_parse == True:
            req_items = self.parse_query_requests(items)
        else:
            req_items = items
        if isinstance(req_items, str):
            error(req_items)
            return None
        data, success = self.send_request(
            self.client,
            "Query/Request/" + self.sn + "/" + self.client_id,
            req_items,
            timeout,
        )
        if not success:
            return None
        values = data["values"]
        if not values or not isinstance(values, list) or len(values) != len(items):
            return None

        value_datas = []

        for e in values:
            if not isinstance(e, dict):
                value_datas.append(None)
                continue

            vv = e.get("values", None)
            if isinstance(vv, list):
                value_datas.append(vv)
            else:
                value_datas.append(None)

        return value_datas


#  .\.env\Scripts\python.exe -m src.core.nc_link
if __name__ == "__main__":
    import time

    broker_ip = "127.0.0.1"
    SN = "1D80F5FB02C08F4"
    port = 1883
    nc_client = NCLink(mqtt_ip=broker_ip, mqtt_port=port, sn=SN, client_id="LUCKDASH")
    nc_client.connect()
    
    # items = {
    #         "values": [
    #             {
    #             "id": "01035437",
    #             "params": {
    #                 "index": 200,
    #                 "value": 1,
    #                 }
    #             },
    #             {
    #             "id": "01035437",
    #             "params": {
    #                 "index": 201,
    #                 "value": 2,
    #                 }
    #             },
    #             ]
    #         }
    

    # TODO: 测试设值
    nc_rev = nc_client.set(
        ["REG_B:200:1", "REG_B:201:2"]
    )
    print(nc_rev)

    # 测试读值
    # while True:
    #     nc_rev = nc_client.query(
    #         items=[
    #             "AXIS_2:0-50",
    #         ],
    #         timeout=2000,
    #     )
    #     print(nc_rev)
    #     time.sleep(2)
        
        
