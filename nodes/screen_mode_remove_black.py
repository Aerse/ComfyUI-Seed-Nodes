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
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "screen_mode_process"
    CATEGORY = "ComfyUI-Seed-Nodes"
    DESCRIPTION = "Screen模式去黑底 - 获取RGB三通道数据，按顺序用滤色模式合成去除黑背景"

    def screen_mode_process(self, image):
        """
        使用Screen模式算法去除黑底，按照指定算法实现

        算法流程:
        1. 从源图获取R,G,B三个通道各自的图像数据
        2. 每个通道数据都带alpha通道(使用通道值作为alpha)
        3. 按照R,G,B顺序用滤色模式合成
        4. 最终得到去掉黑背景的透明PNG

        参数:
            image (torch.Tensor): 输入图像张量

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
            processed_img = self._apply_screen_mode(img_data)
            
            # 转换回张量
            processed_tensor = torch.from_numpy(processed_img).to(device, dtype=dtype)
            results.append(processed_tensor)
        
        # 堆叠批次结果
        return (torch.stack(results, dim=0),)

    def _apply_screen_mode(self, img_data):
        """
        应用Screen模式核心算法，按照用户描述的算法实现

        参数:
            img_data: 图像数据数组 (H, W, C)

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
        
        # 按照用户算法：
        # 1. 获取R,G,B三个通道各自的图像数据，这些数据要带alpha通道
        
        # R通道图像数据(带alpha)
        r_data = np.zeros((height, width, 4), dtype=np.float32)
        r_data[:,:,0] = red      # R通道
        r_data[:,:,1] = 0        # G=0
        r_data[:,:,2] = 0        # B=0  
        r_data[:,:,3] = red      # Alpha使用R通道值
        
        # G通道图像数据(带alpha)
        g_data = np.zeros((height, width, 4), dtype=np.float32)
        g_data[:,:,0] = 0        # R=0
        g_data[:,:,1] = green    # G通道
        g_data[:,:,2] = 0        # B=0
        g_data[:,:,3] = green    # Alpha使用G通道值
        
        # B通道图像数据(带alpha)
        b_data = np.zeros((height, width, 4), dtype=np.float32)
        b_data[:,:,0] = 0        # R=0
        b_data[:,:,1] = 0        # G=0
        b_data[:,:,2] = blue     # B通道
        b_data[:,:,3] = blue     # Alpha使用B通道值
        
        # 2. 按照R,G,B顺序用滤色模式合成
        # 滤色公式: result = 1 - (1 - base) * (1 - blend)
        
        # 从黑色背景开始
        result = np.zeros((height, width, 4), dtype=np.float32)
        
        # 第一步：与R通道数据滤色合成
        result[:,:,0] = 1 - (1 - result[:,:,0]) * (1 - r_data[:,:,0])
        result[:,:,1] = 1 - (1 - result[:,:,1]) * (1 - r_data[:,:,1])
        result[:,:,2] = 1 - (1 - result[:,:,2]) * (1 - r_data[:,:,2])
        # Alpha通道也用滤色模式合成
        result[:,:,3] = 1 - (1 - result[:,:,3]) * (1 - r_data[:,:,3])
        
        # 第二步：与G通道数据滤色合成
        result[:,:,0] = 1 - (1 - result[:,:,0]) * (1 - g_data[:,:,0])
        result[:,:,1] = 1 - (1 - result[:,:,1]) * (1 - g_data[:,:,1])
        result[:,:,2] = 1 - (1 - result[:,:,2]) * (1 - g_data[:,:,2])
        # Alpha通道也用滤色模式合成
        result[:,:,3] = 1 - (1 - result[:,:,3]) * (1 - g_data[:,:,3])
        
        # 第三步：与B通道数据滤色合成
        result[:,:,0] = 1 - (1 - result[:,:,0]) * (1 - b_data[:,:,0])
        result[:,:,1] = 1 - (1 - result[:,:,1]) * (1 - b_data[:,:,1])
        result[:,:,2] = 1 - (1 - result[:,:,2]) * (1 - b_data[:,:,2])
        # Alpha通道也用滤色模式合成
        result[:,:,3] = 1 - (1 - result[:,:,3]) * (1 - b_data[:,:,3])
        
        return result

    def get_info(self):
        """
        返回节点信息

        返回:
            str: 节点的详细信息
        """
        return """
        🎬 Screen模式去黑底节点
        
        算法流程:
        • 从源混黑背景的图上获取R,G,B三个通道各自的图像数据
        • 这些数据一定要带alpha通道(使用通道值作为alpha)
        • 按照R,G,B的顺序用滤色模式合成
        • 最终得到去掉黑背景的透明PNG图
        
        技术特点:
        • 零参数设计，即插即用
        • 使用真实滤色公式 1-(1-base)*(1-blend)
        • Alpha通道也参与滤色合成
        • 完全按照指定算法实现
        
        使用场景:
        • 序列帧特效合成
        • 粒子效果叠加  
        • 光效处理
        • 火焰、爆炸等特效
        """ 