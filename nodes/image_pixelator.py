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

import os
import io
import numpy as np
from PIL import Image
import torch


class ImagePixelatorNode:
    """
    单图像像素化处理节点，实现图像像素化功能
    """

    @staticmethod
    def get_pixelation_params():
        """
        获取像素化参数

        返回:
            dict: 像素化参数，包括必需和可选参数
        """
        params = {
            "required": {
                "pixel_block": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "display": "slider",
                    },
                ),
            },
            "optional": {},
        }
        return params

    @classmethod
    def INPUT_TYPES(cls):
        """
        定义节点的输入类型

        返回:
            dict: 输入参数的定义，包括图像和像素块大小
        """
        params = cls.get_pixelation_params()
        params["required"]["image"] = ("IMAGE",)
        return params

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "pixelate_image_node"
    CATEGORY = "ComfyUI-Seed-Nodes"

    def pixelate_image_node(self, image, pixel_block):
        """
        执行图像像素化处理。

        参数:
            image (torch.Tensor): 输入图像张量。
            pixel_block (int): 像素块大小。

        返回:
            torch.Tensor: 像素化后的图像张量。
        """
        # 确保输入为 torch.Tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        input_image = image.cpu().numpy()

        # 预处理输入图像
        input_image = self.preprocess_input(input_image)

        # 转换为 PIL 图像，保留原始模式（RGB 或 RGBA）
        if input_image.shape[-1] == 4:
            img = Image.fromarray((input_image * 255).astype(np.uint8), "RGBA")
        else:
            img = Image.fromarray((input_image * 255).astype(np.uint8), "RGB")

        # 应用像素化效果
        img = self.pixelate_image(img, pixel_block)

        # 转换回张量
        result = self.convert_to_tensor(img, image.shape, image.device)

        return (result,)

    def pixelate_image(self, img, pixel_block):
        """
        使用最近邻插值算法进行图像像素化。

        参数:
            img (PIL.Image.Image): 输入图像。
            pixel_block (int): 像素块大小。

        返回:
            PIL.Image.Image: 像素化后的图像。
        """
        width, height = img.size
        # 计算缩小后的尺寸，确保至少为1像素
        scaled_width = max(1, width // pixel_block)
        scaled_height = max(1, height // pixel_block)
        # 使用最近邻算法缩小和放大图像
        img = img.resize((scaled_width, scaled_height), Image.Resampling.NEAREST)
        img = img.resize((width, height), Image.Resampling.NEAREST)
        return img

    def preprocess_input(self, input_image):
        """
        预处理输入数组，处理特殊形状和通道顺序。

        参数:
            input_image (numpy.ndarray): 输入图像数组。

        返回:
            numpy.ndarray: 预处理后的图像数组。
        """
        # 处理特殊的 (1, 1, C) 形状
        if (
            len(input_image.shape) == 3
            and input_image.shape[0] == 1
            and input_image.shape[1] == 1
        ):
            channels = input_image.shape[2]
            target_channels = 4 if channels % 4 == 0 else 3
            side_length = int(np.sqrt(channels / target_channels))
            if channels % target_channels:
                side_length += 1

            flat_data = input_image[
                0, 0, : channels - (channels % target_channels)
            ].reshape(-1, target_channels)
            input_image = np.zeros(
                (side_length, side_length, target_channels), dtype=np.float32
            )
            input_image[: flat_data.shape[0] // side_length, :side_length] = (
                flat_data.reshape(-1, side_length, target_channels)
            )

        # 处理 4D 张量（批处理）情况
        elif len(input_image.shape) == 4:
            input_image = input_image[0]

        # 处理通道顺序
        if len(input_image.shape) == 3 and input_image.shape[-1] not in [3, 4]:
            if input_image.shape[0] in [3, 4]:
                input_image = np.transpose(input_image, (1, 2, 0))

        # 如果需要，归一化到 [0, 1] 范围
        if input_image.max() > 1.0:
            input_image = input_image / 255.0

        return input_image

    def convert_to_tensor(self, img, original_shape, device):
        """
        将 PIL 图像转换回张量，正确处理 alpha 通道。

        参数:
            img (PIL.Image.Image): PIL 图像。
            original_shape (tuple): 原始图像的形状。
            device (torch.device): 设备（CPU 或 GPU）。

        返回:
            torch.Tensor: 转换后的图像张量。
        """
        # 检查原始图像是否有 alpha 通道
        has_alpha = len(original_shape) > 2 and original_shape[-1] == 4

        if has_alpha:
            # 确保图像模式为 RGBA
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            compressed_array = np.array(img).astype(np.float32) / 255.0
        else:
            img = img.convert("RGB")
            compressed_array = np.array(img).astype(np.float32) / 255.0
            if len(compressed_array.shape) == 2:
                compressed_array = np.stack([compressed_array] * 3, axis=-1)

        # 处理批处理维度
        if len(original_shape) == 4:
            compressed_array = np.expand_dims(compressed_array, 0)

        return torch.from_numpy(compressed_array).to(device)
