import json
import random
import paho.mqtt.client as mqtt  # 1.6.1

# MQTT服务器地址
MQTT_SERVER = "localhost"
# 发布的主题
PUBLISH_TOPIC_QUERY = "Query/Response/1D80F5FB02C08F4"
PUBLISH_TOPIC_SET = "Set/Response/1D80F5FB02C08F4"
# 实际的查询响应
REAL_QUERY_RESPONSE = {"values":
                    [
                        {"values":[238,259,255,258,250,256,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"code":"OK","id":"01035433","params":{"indexes":["3080-3100"]}},
                        {"values":[27.60991,0.00091,27.60991,27.609,0],"code":"OK","id":"01035455","params":{"indexes":["38","41","43","49","53"]}},
                        {"values":[-145.78152,0.00248,-145.78152,-145.784,0],"code":"OK","id":"01035456","params":{"indexes":["38","41","43","49","53"]}},
                        {"values":[-247.161,50,-247.161,-297.161,0],"code":"OK","id":"01035457","params":{"indexes":["38","41","43","49","53"]}},
                        {"values":[1,9,[500,0,0,0]],"code":"OK","id":"01035450","params":{"indexes":["27","32","47"]}}
                    ]
                }

def get_publish_message_query():
    # 原始的 values 列表
    original_values = [238, 259, 255, 258, 250, 256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # 生成带有随机波动的 values 列表
    fluctuated_values = [
        value + random.randint(-10, 30)  #添加随机波动
        for value in original_values
    ]

    orin_rpm = 5000
    fluctuated_rpm = orin_rpm + random.randint(-2000, 5000)
    
    return {
        "values": [
            {
                "values": fluctuated_values,
                "code": "OK",
                "id": "01035433",
                "params": {"indexes": ["3080-3100"]}
            },
            {
                "values": [27.60991, 0.00091, 27.60991, 27.609, 0],
                "code": "OK",
                "id": "01035455",
                "params": {"indexes": ["40", "41", "43", "49", "53"]}
            },
            {
                "values": [-145.78152, 0.00248, -145.78152, -145.784, 0],
                "code": "OK",
                "id": "01035456",
                "params": {"indexes": ["40", "41", "43", "49", "53"]}
            },
            {
                "values": [-247.161, 50, -247.161, -297.161, 0],
                "code": "OK",
                "id": "01035457",
                "params": {"indexes": ["40", "41", "43", "49", "53"]}
            },
            {
                "values": [1, 9, [fluctuated_rpm, 0, 0, 0]],
                "code": "OK",
                "id": "01035450",
                "params": {"indexes": ["27", "32", "47"]}
            }
        ]
    }

# 当连接到MQTT服务器时的回调
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # 订阅主题
    client.subscribe("Query/Request/1D80F5FB02C08F4/#")
    client.subscribe("Set/Request/1D80F5FB02C08F4/#")
    

# 当接收到订阅主题的消息时的回调
def on_message(client, userdata, msg):
    # print(f"Received message: {msg.payload} on topic {msg.topic}")
    # 尝试解析接收到的消息为JSON
    try:
        message = json.loads(msg.payload.decode('utf-8'))
        print(f"Message parsed as JSON: {message}")
    except json.JSONDecodeError:
        print("Error: Received message is not a valid JSON string")
    
    if msg.topic.startswith("Query/Request/1D80F5FB02C08F4"):
        # 从主题中提取client_id
        topic_parts = msg.topic.split('/')
        # 确保主题格式正确，避免索引错误
        client_id = topic_parts[3]  # 假设client_id位于第4部分
        # 将要发布的消息转换为JSON格式的字符串
        publish_message = get_publish_message_query()
        publish_message["@id"] = message.get("@id", "default_id")  # 使用get避免KeyError
        message_to_publish_Q = json.dumps(publish_message)
        # 构造包含client_id的发布主题
        publish_topic_with_client_id_Q = f"{PUBLISH_TOPIC_QUERY}/{client_id}"
        # 发布新的消息
        client.publish(publish_topic_with_client_id_Q, message_to_publish_Q)
     
# 创建MQTT客户端实例
client = mqtt.Client()
# 绑定回调函数
client.on_connect = on_connect
client.on_message = on_message

# 连接到MQTT服务器
client.connect(MQTT_SERVER, 1883, 60)

# 阻塞循环，直到手动停止执行
client.loop_forever()
