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
import hashlib
import torch
import numpy as np
from PIL import Image, ImageSequence, ImageOps

import folder_paths
import node_helpers


class LoadImage:
    """
    一个用于加载单张图像的类。
    """

    CATEGORY = "ComfyUI-Seed-Nodes"  # 类别名称
    DESCRIPTION = "加载图像"  # 类描述
    FUNCTION = "load_image"  # 函数名称
    RETURN_TYPES = ("IMAGE", "MASK")  # 返回类型
    RETURN_NAMES = ("image", "mask")  # 返回名称

    def __init__(self):
        pass  # 初始化方法，目前无操作

    @classmethod
    def INPUT_TYPES(cls):
        """
        定义输入类型，包括必需和可选参数。

        返回:
            dict: 输入参数的定义，包括图像文件列表和是否保留alpha通道的选项。
        """
        input_dir = folder_paths.get_input_directory()  # 获取输入目录路径
        files = sorted(
            [
                f
                for f in os.listdir(input_dir)
                if os.path.isfile(os.path.join(input_dir, f))
            ]
        )  # 获取输入目录中的所有文件，并按名称排序

        return {
            "required": {
                "image": (files, {"image_upload": True}),  # 必需参数：图像文件列表
                "keep_alpha_channel": (
                    "BOOLEAN",
                    {"default": True, "label_on": "enabled", "label_off": "disabled"},
                ),  # 可选参数：是否保留alpha通道
            },
        }

    @staticmethod
    def load_image(image, keep_alpha_channel):
        """
        加载图像及其对应的遮罩。

        参数:
            image (str): 图像文件名。
            keep_alpha_channel (bool): 是否保留alpha通道。

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 返回加载的图像张量和遮罩张量。
        """
        image_path = folder_paths.get_annotated_filepath(image)  # 获取图像的完整路径
        img = node_helpers.pillow(Image.open, image_path)  # 使用自定义助手函数打开图像

        output_images = []  # 存储处理后的图像张量
        output_masks = []  # 存储对应的遮罩张量
        width, height = None, None  # 初始化图像宽度和高度
        excluded_formats = {"MPO"}  # 排除的图像格式

        # 遍历图像的所有帧（适用于多帧图像，如GIF）
        for frame in ImageSequence.Iterator(img):
            frame = node_helpers.pillow(
                ImageOps.exif_transpose, frame
            )  # 自动旋转图像以纠正EXIF信息

            if frame.mode == "I":
                # 如果图像模式为 'I'（32位整型像素），则进行归一化处理
                frame = frame.point(lambda pixel: pixel * (1 / 255))

            has_alpha = "A" in frame.getbands()  # 检查图像是否有alpha通道
            if has_alpha and keep_alpha_channel:
                processed_image = frame.convert("RGBA")  # 保留alpha通道
            else:
                processed_image = frame.convert("RGB")  # 转换为RGB模式

            if not output_images:
                width, height = processed_image.size  # 获取图像尺寸

            if processed_image.size != (width, height):
                continue  # 跳过尺寸不一致的帧

            image_array = (
                np.array(processed_image).astype(np.float32) / 255.0
            )  # 将图像转换为NumPy数组并归一化
            image_tensor = torch.from_numpy(image_array).unsqueeze(
                0
            )  # 转换为Torch张量，并增加一个批次维度

            if has_alpha:
                alpha_channel = (
                    np.array(frame.getchannel("A")).astype(np.float32) / 255.0
                )  # 获取alpha通道并归一化
                mask_tensor = 1.0 - torch.from_numpy(alpha_channel).unsqueeze(
                    0
                )  # 创建遮罩张量，并反转
            else:
                mask_tensor = torch.zeros(
                    (1, 64, 64), dtype=torch.float32
                )  # 如果没有alpha通道，创建一个固定尺寸的零遮罩

            output_images.append(image_tensor)  # 添加图像张量到列表
            output_masks.append(mask_tensor)  # 添加遮罩张量到列表

        # 如果有多帧且图像格式不在排除列表中，则将所有帧的张量拼接在一起
        if len(output_images) > 1 and img.format not in excluded_formats:
            final_image = torch.cat(output_images, dim=0)  # 拼接图像张量
            final_mask = torch.cat(output_masks, dim=0)  # 拼接遮罩张量
        else:
            final_image = output_images[0]  # 单帧图像直接取第一张
            final_mask = output_masks[0]  # 单帧遮罩直接取第一张

        return final_image, final_mask  # 返回最终的图像和遮罩张量

    @classmethod
    def IS_CHANGED(cls, image):
        """
        检查图像是否发生变化，通过计算文件的SHA256哈希值。

        参数:
            image (str): 图像文件名。

        返回:
            str: 图像文件的SHA256哈希值。
        """
        image_path = folder_paths.get_annotated_filepath(image)  # 获取图像的完整路径
        sha256_hash = hashlib.sha256()  # 创建SHA256哈希对象
        with open(image_path, "rb") as file:
            sha256_hash.update(file.read())  # 读取文件内容并更新哈希对象
        return sha256_hash.hexdigest()  # 返回哈希值的十六进制表示

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        """
        验证输入的图像文件是否存在。

        参数:
            image (str): 图像文件名。

        返回:
            Union[bool, str]: 如果验证通过，返回True；否则，返回错误信息。
        """
        if not folder_paths.exists_annotated_filepath(image):
            return f"LoadImage: 无效的图像文件: {image}"  # 返回错误信息
        return True  # 验证通过
