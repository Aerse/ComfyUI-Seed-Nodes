# MIT License
# 
# Copyright (c) 2024 Hmily
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

# 修改自 ComfyUI-Light-Tool
# 修改者: Seed
# 修改日期: 2024-12-05

import os
import torch
import numpy as np
from PIL import Image, ImageOps
from typing import List, Tuple


class LoadMultipleImages:
    """
    一个用于加载多个图像的类。
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        定义输入类型，包括必需和可选参数。

        返回:
            dict: 输入参数的定义，包括图像目录路径和是否保留alpha通道的选项。
        """
        return {
            "required": {
                "directory": ("STRING", {"default": "please input your image dir path"}),
            },
            "optional": {
                "keep_alpha_channel": (
                    "BOOLEAN",
                    {"default": True, "label_on": "enabled", "label_off": "disabled"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "load_images"
    CATEGORY = "ComfyUI-Seed-Nodes"
    DESCRIPTION = "Load image From image directory"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        判断输入参数是否发生变化。

        参数:
            **kwargs: 任意关键字参数。

        返回:
            int: 输入参数的哈希值。
        """
        return hash(frozenset(kwargs))

    @staticmethod
    def load_images(directory: str, keep_alpha_channel: bool = False) -> Tuple[List, List]:
        """
        加载多个图像及其对应的遮罩。

        参数:
            directory (str): 图像目录路径。
            keep_alpha_channel (bool): 是否保留alpha通道。

        返回:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: 返回加载的图像张量列表和遮罩张量列表。
        """
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)
        file_paths = [os.path.join(directory, x) for x in dir_files]

        from itertools import islice
        file_paths = list(
            islice(file_paths, 0, None))  # 这里可以调整加载的文件数量

        images, masks = [], []
        for image_path in file_paths:
            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img)  # 自动旋转图像以纠正EXIF信息
                has_alpha = "A" in img.getbands()
                if has_alpha and keep_alpha_channel:
                    image = img.convert("RGBA")
                else:
                    image = img.convert("RGB")

                image_array = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_array).unsqueeze(0)  # 增加批次维度

                if 'A' in img.getbands():
                    mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                    mask_tensor = 1.0 - torch.from_numpy(mask)  # 创建遮罩张量并反转
                else:
                    mask_tensor = torch.zeros((64, 64), dtype=torch.float32)  # 固定尺寸的零遮罩

                images.append(image_tensor)
                masks.append(mask_tensor)

        return images, masks
