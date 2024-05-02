import serial
import time

# 配置串口参数
ser = serial.Serial(
    port='/dev/ttyAMA10',    # 树莓派的串口设备，需要根据你的设置进行更改
    baudrate=9600,          # 配置波特率，和你的单片机保持一致
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1               # 读取超时时间
)

# 发送数据
def send_data(data):
    ser.write(data)

# 接收数据
def receive_data():
    while True:
        if ser.in_waiting:
            data = ser.read(ser.in_waiting)
            return data

# 主函数
def main():
    try:
        while True:
            # 例子：发送 "Hello" 到单片机
            send_data(b'Hello\n')
            
            # 等待一段时间
            time.sleep(1)
            
            # 例子：接收来自单片机的数据
            incoming_data = receive_data()
            if incoming_data:
                print("Received:", incoming_data)
            
            # 延时
            time.sleep(1)
    except KeyboardInterrupt:
        ser.close()

if __name__ == '__main__':
    main()