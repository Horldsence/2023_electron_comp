import serial
import struct

# 定义串口设置
ser = serial.Serial(
    port="/dev/ttyAMA10",
    baudrate=9600,
    timeout=1
)

# 假设坐标点是这样的一个列表，每个点是一个二元组(x, y)
points = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]

# 计算简单校验和（加和并且模256以适应单字节）
def calculate_checksum(data):
    return sum(data) % 256

# 打包数据和校验和
data = struct.pack('<' + 'H' * len(points) * 2, *sum(points, ()))
checksum = calculate_checksum(data)
# b'\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00\x07\x00\x08\x00\t\x00\n\x00\x0b\x00\x0c\x00'
# b'\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00\x07\x00\x08\x00\t\x00\n\x00\x0b\x00\x0c\x00N'
packed_data_with_checksum = data + struct.pack('<B', checksum)

print(packed_data_with_checksum)
# 发送数据包加上校验和
ser.write(packed_data_with_checksum)

# 关闭串口
ser.close()