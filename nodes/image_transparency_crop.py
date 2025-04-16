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
from PIL import Image
import torch


class ImageTransparencyCropNode:
    """
    图像透明区域裁剪节点，用于裁剪图像的透明区域，只保留非透明部分
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        定义节点的输入类型

        返回:
            dict: 输入参数的定义，包括图像
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "padding": (
                    "INT", 
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "display": "slider",
                    }
                ),
                "threshold": (
                    "FLOAT", 
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                    }
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "crop_transparency"
    CATEGORY = "ComfyUI-Seed-Nodes"
    DESCRIPTION = "裁剪图像的透明区域，只保留非透明部分"

    def crop_transparency(self, image, padding=0, threshold=0.0):
        """
        裁剪图像中的透明区域，只保留非透明部分。

        参数:
            image (torch.Tensor): 输入图像张量。
            padding (int): 在裁剪区域周围添加的额外边距（像素）。
            threshold (float): 透明度阈值，低于此值的像素被视为透明（范围：0.0-1.0）。

        返回:
            torch.Tensor: 裁剪后的图像张量。
        """
        # 确保输入为 torch.Tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        
        # 保存原始设备和批次信息
        device = image.device
        batch_size = image.shape[0]
        results = []
        
        # 逐批次处理图像
        for b in range(batch_size):
            img_data = image[b].cpu().numpy()
            
            # 转换为PIL图像以便处理
            if img_data.shape[2] == 4:
                # 已经有RGBA通道
                pil_img = Image.fromarray((img_data * 255).astype(np.uint8), "RGBA")
            elif img_data.shape[2] == 3:
                # 没有Alpha通道，无法进行透明度裁剪，直接返回原图
                pil_img = Image.fromarray((img_data * 255).astype(np.uint8), "RGB")
                results.append(image[b])
                continue
            
            # 获取Alpha通道并创建掩码
            alpha = np.array(pil_img.getchannel('A'))
            alpha_mask = alpha > (threshold * 255)  # 阈值过滤
            
            if not np.any(alpha_mask):
                # 如果整个图像都是透明的，直接返回原图
                results.append(image[b])
                continue
            
            # 找到非透明区域的边界框
            rows = np.any(alpha_mask, axis=1)
            cols = np.any(alpha_mask, axis=0)
            
            if not np.any(rows) or not np.any(cols):
                # 如果整个图像都是透明的，直接返回原图
                results.append(image[b])
                continue
            
            # 获取非透明区域的边界
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            
            # 应用边距（padding）
            height, width = pil_img.height, pil_img.width
            ymin = max(0, ymin - padding)
            ymax = min(height - 1, ymax + padding)
            xmin = max(0, xmin - padding)
            xmax = min(width - 1, xmax + padding)
            
            # 裁剪图像
            cropped_img = pil_img.crop((xmin, ymin, xmax + 1, ymax + 1))
            
            # 转换回张量
            if cropped_img.mode == "RGBA":
                cropped_array = np.array(cropped_img).astype(np.float32) / 255.0
            else:
                cropped_array = np.array(cropped_img.convert("RGBA")).astype(np.float32) / 255.0
            
            cropped_tensor = torch.from_numpy(cropped_array).to(device)
            results.append(cropped_tensor)
        
        # 处理结果
        if batch_size == 1:
            # 单张图像情况
            return (results[0].unsqueeze(0),)
        else:
            # 多张图像情况
            # 检查所有张量尺寸是否一致
            shapes = [tensor.shape for tensor in results]
            if all(shape == shapes[0] for shape in shapes):
                # 所有裁剪后的图像尺寸相同，可以直接堆叠
                return (torch.stack(results, dim=0),)
            else:
                # 尺寸不同，需要单独处理每张图像
                # 在实际应用中，你可能需要根据具体需求决定如何处理这种情况
                # 这里简单起见，只返回第一张图像
                return (results[0].unsqueeze(0),) 