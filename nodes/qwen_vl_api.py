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
import json
import base64
import torch
import requests
import numpy as np
from PIL import Image
import folder_paths


class QwenVLAPINode:
    """
    通义千问VL API图像理解节点。
    """

    CATEGORY = "ComfyUI-Seed-Nodes"  # 类别名称
    DESCRIPTION = "通义千问视觉理解"  # 类描述
    FUNCTION = "process_image"  # 函数名称
    RETURN_TYPES = ("STRING",)  # 返回类型
    RETURN_NAMES = ("response",)  # 返回名称

    @classmethod
    def INPUT_TYPES(cls):
        """
        定义输入类型，包括必需和可选参数。

        返回:
            dict: 输入参数的定义
        """
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "请描述这张图片的内容", "multiline": True}),
                "model": (
                    [
                        "qwen-vl-max",
                        "qwen-vl-max-latest",
                        "qwen-vl-max-2025-04-08",
                        "qwen-vl-max-2025-04-02",
                        "qwen-vl-plus",
                        "qwen-vl-plus-latest",
                        "qwen-vl-plus-2025-01-25"
                    ],
                    {"default": "qwen-vl-plus"}
                ),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    def process_image(self, api_key, prompt, model, image=None):
        """
        处理图像并通过通义千问API获取响应。

        参数:
            api_key (str): 通义千问API密钥
            prompt (str): 提示词
            model (str): 模型名称
            image (torch.Tensor, optional): 输入图像, 形状为 [batch_size, height, width, channels]

        返回:
            Tuple[str]: 返回API的响应文本
        """
        if not api_key:
            return ("API密钥不能为空，请提供有效的通义千问API密钥。",)

        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # 构建请求消息
        messages = [
            {
                "role": "system",
                "content": [
                    {"text": "You are a helpful assistant."}
                ]
            },
            {
                "role": "user",
                "content": []
            }
        ]

        # 如果提供了图像，则添加到用户消息中
        if image is not None:
            # 选择第一个批次的图像
            if len(image.shape) == 4:
                image_tensor = image[0]
            else:
                image_tensor = image
            
            # 将图像转换为PIL格式
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            
            # 转换为Base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            # 添加图像到用户消息
            messages[1]["content"].append({"image": f"data:image/png;base64,{base64_image}"})
        
        # 添加提示词到用户消息
        messages[1]["content"].append({"text": prompt})

        # 构建请求数据
        data = {
            "model": model,
            "input": {
                "messages": messages
            }
        }

        try:
            # 发送API请求
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            
            # 提取文本响应
            if "output" in result and "choices" in result["output"] and len(result["output"]["choices"]) > 0:
                message = result["output"]["choices"][0]["message"]
                if "content" in message and len(message["content"]) > 0:
                    for content_item in message["content"]:
                        if "text" in content_item:
                            return (content_item["text"],)
            
            # 如果没有找到文本响应，返回完整的JSON响应
            return (json.dumps(result, ensure_ascii=False, indent=2),)
            
        except Exception as e:
            return (f"API请求出错: {str(e)}",) 