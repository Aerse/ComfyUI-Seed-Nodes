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
    图像透明区域裁剪节点，用于裁剪图像的透明区域，只保留非透明部分（改进版）
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
                "batch_strategy": (
                    ["first_only", "largest_size", "most_common", "individual"], 
                    {
                        "default": "individual"
                    }
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "crop_transparency"
    CATEGORY = "ComfyUI-Seed-Nodes"
    DESCRIPTION = "裁剪图像的透明区域，只保留非透明部分（改进版 - 修复批次处理问题）"

    def crop_transparency(self, image, padding=0, threshold=0.0, batch_strategy="individual"):
        """
        裁剪图像中的透明区域，只保留非透明部分。

        参数:
            image (torch.Tensor): 输入图像张量。
            padding (int): 在裁剪区域周围添加的额外边距（像素）。
            threshold (float): 透明度阈值，低于此值的像素被视为透明（范围：0.0-1.0）。
            batch_strategy (str): 批次处理策略。

        返回:
            torch.Tensor: 裁剪后的图像张量。
        """
        # 确保输入为 torch.Tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        
        # 保存原始设备和数据类型信息
        device = image.device
        dtype = image.dtype
        batch_size = image.shape[0]
        results = []
        
        # 确保在CPU上处理以避免设备不一致问题
        image_cpu = image.cpu()
        
        # 逐批次处理图像
        for b in range(batch_size):
            img_data = image_cpu[b].numpy()
            
            # 确保数据在正确范围内并转换为PIL图像
            if img_data.shape[2] == 4:
                # 已经有RGBA通道
                # 修复：确保数据范围正确，避免精度问题
                img_data_255 = np.clip(img_data * 255.0, 0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img_data_255, "RGBA")
            elif img_data.shape[2] == 3:
                # 没有Alpha通道，为RGB图像添加一个全不透明的Alpha通道
                img_data_255 = np.clip(img_data * 255.0, 0, 255).astype(np.uint8)
                # 添加Alpha通道
                alpha_channel = np.full((img_data_255.shape[0], img_data_255.shape[1], 1), 255, dtype=np.uint8)
                rgba_data = np.concatenate([img_data_255, alpha_channel], axis=2)
                pil_img = Image.fromarray(rgba_data, "RGBA")
            else:
                # 不支持的通道数，直接返回原图
                results.append(image[b])
                continue
            
            # 获取Alpha通道并创建掩码
            alpha = np.array(pil_img.getchannel('A'))
            # 修复：使用更稳定的阈值比较
            alpha_mask = alpha >= (threshold * 255.0)  # 使用 >= 而不是 > 提高稳定性
            
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
            
            # 转换回张量，确保精度一致性
            cropped_array = np.array(cropped_img).astype(np.float32) / 255.0
            cropped_tensor = torch.from_numpy(cropped_array).to(device, dtype=dtype)
            results.append(cropped_tensor)
        
        # 修复：改进的批次处理逻辑
        return self._handle_batch_results(results, batch_strategy)
    
    def _handle_batch_results(self, results, batch_strategy):
        """
        处理批次结果的不同策略
        
        参数:
            results: 处理结果列表
            batch_strategy: 批次处理策略
            
        返回:
            处理后的张量
        """
        if len(results) == 1:
            # 单张图像情况
            return (results[0].unsqueeze(0),)
        
        # 分析所有结果的形状
        shapes = [tensor.shape for tensor in results]
        unique_shapes = list(set([str(shape) for shape in shapes]))
        
        if len(unique_shapes) == 1:
            # 所有图像尺寸相同，可以直接堆叠
            return (torch.stack(results, dim=0),)
        
        # 尺寸不同的情况，根据策略处理
        if batch_strategy == "first_only":
            # 只返回第一张图像（原始行为）
            return (results[0].unsqueeze(0),)
        
        elif batch_strategy == "largest_size":
            # 返回最大尺寸的图像
            sizes = [tensor.shape[0] * tensor.shape[1] for tensor in results]
            largest_idx = sizes.index(max(sizes))
            return (results[largest_idx].unsqueeze(0),)
        
        elif batch_strategy == "most_common":
            # 返回最常见尺寸的所有图像
            shape_counts = {}
            for shape in shapes:
                shape_key = str(shape)
                shape_counts[shape_key] = shape_counts.get(shape_key, 0) + 1
            
            # 找到最常见的尺寸
            most_common_shape_str = max(shape_counts.items(), key=lambda x: x[1])[0]
            most_common_shape = eval(most_common_shape_str)
            
            # 收集具有最常见尺寸的图像
            filtered_results = []
            for tensor in results:
                if tensor.shape == most_common_shape:
                    filtered_results.append(tensor)
            
            if filtered_results:
                return (torch.stack(filtered_results, dim=0),)
            else:
                return (results[0].unsqueeze(0),)
        
        elif batch_strategy == "individual":
            # 返回所有图像，但分别处理（推荐策略）
            # 由于ComfyUI限制，我们选择返回第一张，但在日志中说明
            print(f"警告: 批次中有 {len(results)} 张图像，尺寸不同。返回第一张图像。")
            print(f"图像尺寸: {[tensor.shape for tensor in results]}")
            return (results[0].unsqueeze(0),)
        
        # 默认返回第一张
        return (results[0].unsqueeze(0),)


# 用于一致性测试的函数
def test_consistency_improved():
    """
    测试改进版本的一致性
    """
    # 创建测试图像
    test_image = torch.rand(1, 64, 64, 4)
    # 设置固定的透明区域
    test_image[0, :10, :, 3] = 0.0    # 顶部透明
    test_image[0, -10:, :, 3] = 0.0   # 底部透明
    test_image[0, :, :10, 3] = 0.0    # 左侧透明
    test_image[0, :, -10:, 3] = 0.0   # 右侧透明
    
    # 设置中间区域为不透明
    test_image[0, 10:-10, 10:-10, 3] = 1.0
    
    node = ImageTransparencyCropNode()
    
    print("测试改进版本的一致性...")
    print(f"输入图像形状: {test_image.shape}")
    
    # 多次执行相同操作
    results = []
    for i in range(20):
        result = node.crop_transparency(test_image, padding=0, threshold=0.13)
        results.append(result[0].shape)
        if i == 0:
            first_result = result[0]
    
    # 检查结果是否一致
    first_shape = results[0]
    all_same = all(shape == first_shape for shape in results)
    
    print(f"一致性测试结果: {'✅ 通过' if all_same else '❌ 失败'}")
    print(f"输出形状: {first_shape}")
    print(f"测试次数: {len(results)}")
    
    if not all_same:
        unique_shapes = list(set([str(shape) for shape in results]))
        print(f"发现的不同形状: {unique_shapes}")
    
    # 检查裁剪是否正确
    expected_crop_size = (1, 44, 44, 4)  # 64-10-10 = 44
    actual_crop_size = first_result.shape
    crop_correct = actual_crop_size == expected_crop_size
    
    print(f"裁剪正确性: {'✅ 正确' if crop_correct else '❌ 错误'}")
    print(f"预期尺寸: {expected_crop_size}")
    print(f"实际尺寸: {actual_crop_size}")
    
    return all_same and crop_correct


if __name__ == "__main__":
    # 运行一致性测试
    test_consistency_improved() 