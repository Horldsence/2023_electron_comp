from picamera2 import Picamera2, Controls
from libcamera import controls


class Imget:
    def __init__(self):
        # 创建一个Picamera2对象的实例
        self.cam = Picamera2()

        # 设置相机预览的分辨率
        self.cam.preview_configuration.main.size = (640, 360)
        self.cam.preview_configuration.main.format = "RGB888"
        # 设置预览帧率
        self.cam.preview_configuration.controls.FrameRate = 50
        # 对预览帧进行校准
        self.cam.preview_configuration.align()
        # 配置相机为预览模式
        self.cam.configure("preview")
        # 设置相机控制参数为连续对焦模式(自动对焦)
        self.cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})

        # 启动相机
        self.cam.start()

    def set_exposure(self, auto_exposure, exposure_time):
        """
        设置曝光模式。
        """
        if auto_exposure:
            # 设置为自动曝光模式
            self.cam.set_controls({"AeEnable": True})
            controls = {"ExposureTime": exposure_time}
        else:
            # 关闭自动曝光
            self.set_exposure_handler(exposure_time)

    def set_exposure_handler(self, exposure_time):
        """
        设置手动曝光时间和ISO。
        :param exposure_time: 曝光时间，单位为微秒
        :param iso: ISO值，传入0则保持当前ISO不变
        """
        self.set_auto_exposure(False)  # 关闭自动曝光
        controls = {"ExposureTime": exposure_time}
        self.cam.set_controls(controls)

    def getImg(self):
        # 获取相机捕获的图像数组(numpy数组)
        frame = self.cam.capture_array()
        # 返回捕获的图像数组
        return frame

    def __del__(self):
        self.cam.stop()
        self.cam.close()
