# MIT License
# 
# Copyright (c) 2024 Seed
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch


class ScreenModeRemoveBlackNode:
    """
    Screen模式去黑底节点，模拟Photoshop滤色叠加效果
    将黑色区域转换为透明渐变，产生自然的过渡效果
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        定义节点的输入类型

        返回:
            dict: 输入参数的定义
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness_boost": (
                    "FLOAT", 
                    {
                        "default": 1.3,
                        "min": 1.0,
                        "max": 2.0,
                        "step": 0.1,
                        "display": "slider",
                    }
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "screen_mode_process"
    CATEGORY = "ComfyUI-Seed-Nodes"
    DESCRIPTION = "Screen模式去黑底 - 模拟Photoshop滤色叠加效果，将黑底转换为透明渐变"

    def screen_mode_process(self, image, brightness_boost=1.3):
        """
        使用Screen模式算法去除黑底，模拟Photoshop滤色效果

        参数:
            image (torch.Tensor): 输入图像张量
            brightness_boost (float): 亮度增强倍数 (1.0-2.0)

        返回:
            tuple: 处理后的图像张量
        """
        # 确保输入为 torch.Tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        
        # 保存原始设备和数据类型信息
        device = image.device
        dtype = image.dtype
        batch_size = image.shape[0]
        
        # 确保在CPU上处理
        image_cpu = image.cpu().float()
        results = []
        
        # 逐批次处理图像
        for b in range(batch_size):
            img_data = image_cpu[b].numpy()
            processed_img = self._apply_screen_mode(img_data, brightness_boost)
            
            # 转换回张量
            processed_tensor = torch.from_numpy(processed_img).to(device, dtype=dtype)
            results.append(processed_tensor)
        
        # 堆叠批次结果
        return (torch.stack(results, dim=0),)

    def _apply_screen_mode(self, img_data, brightness_boost):
        """
        应用Screen模式核心算法

        参数:
            img_data: 图像数据数组 (H, W, C)
            brightness_boost: 亮度增强倍数

        返回:
            numpy.ndarray: 处理后的RGBA图像数据
        """
        height, width, channels = img_data.shape
        
        # 处理不同通道数的图像
        if channels == 3:
            # RGB图像，添加Alpha通道
            alpha_channel = np.ones((height, width, 1), dtype=np.float32)
            img_data = np.concatenate([img_data, alpha_channel], axis=2)
        elif channels == 4:
            # RGBA图像，直接使用
            pass
        else:
            # 不支持的通道数，转换为RGB+Alpha
            if channels == 1:
                # 灰度图像
                img_data = np.repeat(img_data, 3, axis=2)
            else:
                # 其他情况，取前3个通道
                img_data = img_data[:, :, :3]
            alpha_channel = np.ones((height, width, 1), dtype=np.float32)
            img_data = np.concatenate([img_data, alpha_channel], axis=2)

        # 分离RGBA通道
        red, green, blue, alpha = img_data[:,:,0], img_data[:,:,1], img_data[:,:,2], img_data[:,:,3]
        
        # 计算亮度值 (使用标准亮度公式)
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        
        # Screen模式核心算法
        # 基于亮度创建平滑的alpha通道，实现渐变透明效果
        enhanced_luminance = np.power(luminance, 0.8)  # 增强对比度
        new_alpha = enhanced_luminance
        
        # 增强颜色亮度
        red = np.clip(red * brightness_boost, 0, 1)
        green = np.clip(green * brightness_boost, 0, 1)
        blue = np.clip(blue * brightness_boost, 0, 1)
        
        # 组合处理结果
        result = np.zeros((height, width, 4), dtype=np.float32)
        result[:,:,0] = red
        result[:,:,1] = green
        result[:,:,2] = blue
        result[:,:,3] = new_alpha
        
        return result

    def get_info(self):
        """
        返回节点信息

        返回:
            str: 节点的详细信息
        """
        return """
        🎬 Screen模式去黑底节点
        
        功能说明:
        • 模拟Photoshop的滤色(Screen)叠加模式
        • 将黑色区域转换为透明渐变
        • 产生自然的过渡效果，适合叠加合成
        
        参数说明:
        • brightness_boost: 亮度增强倍数，提升整体亮度
        
        使用场景:
        • 序列帧特效合成
        • 粒子效果叠加
        • 光效处理
        • 火焰、爆炸等特效
        """ 