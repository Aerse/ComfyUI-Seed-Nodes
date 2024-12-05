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
import cv2


class SLIC:
    """
    SLIC算法实现，用于图像的超像素分割。
    """

    def __init__(self, image, step=10, iters=10, stride=10, weight=10):
        """
        初始化 SLIC 对象。

        参数:
            image (numpy.ndarray): 输入图像，格式为 (H, W, C)，BGR 颜色空间。
            step (int): 网格间隔（步长），控制聚类中心的间隔。
            iters (int): 迭代次数，控制聚类的精度。
            stride (int): 像素化的步幅，控制最终图像的像素块大小。
            weight (float): 颜色距离的权重，控制颜色相似性的影响程度。
        """
        self.image = image
        self.height, self.width, _ = image.shape
        self.step = step
        self.iters = iters
        self.stride = stride
        self.weight = weight

        # 转换到 LAB 颜色空间
        self.lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB).astype(np.float64)

        # 初始化聚类分配和距离映射
        self.clusterID = -1 * np.ones((self.height, self.width), dtype=np.int32)
        self.distances = np.full((self.height, self.width), np.inf)

        # 初始化聚类中心
        self.centers = []
        self.initialize_centers()

    def initialize_centers(self):
        """
        在规则网格上初始化聚类中心，并调整到局部最小梯度。
        """
        S = self.step
        for y in range(S // 2, self.height, S):
            for x in range(S // 2, self.width, S):
                # 在 3x3 邻域中找到最小梯度
                min_grad = np.inf
                loc_min = (y, x)
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny = y + dy
                        nx = x + dx
                        if 0 <= ny < self.height - 1 and 0 <= nx < self.width - 1:
                            grad = np.sum(np.abs(self.lab[ny, nx] - self.lab[ny, nx + 1])) + \
                                   np.sum(np.abs(self.lab[ny, nx] - self.lab[ny + 1, nx]))
                            if grad < min_grad:
                                min_grad = grad
                                loc_min = (ny, nx)
                L, A, B = self.lab[loc_min[0], loc_min[1]]
                self.centers.append([L, A, B, loc_min[0], loc_min[1]])
        self.centers = np.array(self.centers)

    def compute_pixel(self):
        """
        执行迭代聚类过程，分配像素到聚类中心。
        """
        for itr in range(self.iters):
            for idx, center in enumerate(self.centers):
                L, A, B, cy, cx = center
                y_min = max(int(cy - self.step), 0)
                y_max = min(int(cy + self.step), self.height)
                x_min = max(int(cx - self.step), 0)
                x_max = min(int(cx + self.step), self.width)

                # 向量化计算邻域内的距离
                ys, xs = np.meshgrid(np.arange(y_min, y_max), np.arange(x_min, x_max), indexing='ij')
                Ls = self.lab[ys, xs, 0]
                As = self.lab[ys, xs, 1]
                Bs = self.lab[ys, xs, 2]

                # 计算 LAB 颜色距离
                d_lab = np.sqrt((L - Ls) ** 2 + (A - As) ** 2 + (B - Bs) ** 2)

                # 计算空间距离
                d_xy = np.sqrt((cy - ys) ** 2 + (cx - xs) ** 2)

                # 组合距离
                d = d_lab / self.weight + d_xy / self.step

                # 更新聚类分配和距离
                mask = d < self.distances[ys, xs]
                self.clusterID[ys[mask], xs[mask]] = idx
                self.distances[ys[mask], xs[mask]] = d[mask]

            # 更新聚类中心
            labels = self.clusterID.flatten()
            # 过滤掉 -1 的标签
            valid = labels >= 0
            labels = labels[valid]
            if labels.size == 0:
                continue

            lab_flat = self.lab.reshape(-1, 3)
            coords = np.indices((self.height, self.width)).reshape(2, -1).T
            L = lab_flat[:, 0]
            A = lab_flat[:, 1]
            B = lab_flat[:, 2]
            y_coords = coords[:, 0]
            x_coords = coords[:, 1]
            num_clusters = len(self.centers)

            # 计算每个聚类的总和
            sum_L = np.bincount(labels, weights=L[valid], minlength=num_clusters)
            sum_A = np.bincount(labels, weights=A[valid], minlength=num_clusters)
            sum_B = np.bincount(labels, weights=B[valid], minlength=num_clusters)
            sum_y = np.bincount(labels, weights=y_coords[valid], minlength=num_clusters)
            sum_x = np.bincount(labels, weights=x_coords[valid], minlength=num_clusters)
            counts = np.bincount(labels, minlength=num_clusters)

            # 避免除以零
            non_zero = counts > 0
            self.centers[non_zero, 0] = sum_L[non_zero] / counts[non_zero]
            self.centers[non_zero, 1] = sum_A[non_zero] / counts[non_zero]
            self.centers[non_zero, 2] = sum_B[non_zero] / counts[non_zero]
            self.centers[non_zero, 3] = sum_y[non_zero] / counts[non_zero]
            self.centers[non_zero, 4] = sum_x[non_zero] / counts[non_zero]

    def pick_pixel(self):
        """
        生成像素化图像，通过将每个块分配为最频繁的聚类中心的颜色。

        返回:
            numpy.ndarray: 像素化后的图像数组。
        """
        row = int(np.ceil(self.height / self.stride))
        col = int(np.ceil(self.width / self.stride))
        result = np.zeros_like(self.image)

        for m in range(row):
            for n in range(col):
                startj = m * self.stride
                startk = n * self.stride
                endj = min(startj + self.stride, self.height)
                endk = min(startk + self.stride, self.width)
                block = self.clusterID[startj:endj, startk:endk]

                # 过滤掉 -1 的标签
                block = block[block >= 0]

                if block.size == 0:
                    continue

                # 找到块内最频繁的聚类ID
                counts = np.bincount(block.flatten())
                centerpos = np.argmax(counts)

                if centerpos != -1 and centerpos < len(self.centers):
                    cy = int(self.centers[centerpos, 3])
                    cx = int(self.centers[centerpos, 4])
                    color = self.image[cy, cx]
                    result[startj:endj, startk:endk] = color

        return result

    def pixel_deal(self):
        """
        执行 SLIC 算法并返回像素化图像。

        返回:
            numpy.ndarray: 像素化后的图像数组。
        """
        self.compute_pixel()
        result = self.pick_pixel()
        return result


class SLICPixelatorNode:
    """
    基于 SLIC 算法的图像像素化处理节点，实现图像像素化功能
    """

    @staticmethod
    def get_slic_params():
        """
        获取 SLIC 像素化参数

        返回:
            dict: SLIC 参数，包括必需和可选参数
        """
        params = {
            "required": {
                "image": ("IMAGE",)
            },
            "optional": {
                "step": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "display": "slider",
                        "label": "SLIC Step",
                    },
                ),
                "iters": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 50,
                        "step": 1,
                        "display": "slider",
                        "label": "SLIC Iterations",
                    },
                ),
                "stride": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "display": "slider",
                        "label": "SLIC Stride",
                    },
                ),
                "weight": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 1.0,
                        "max": 100.0,
                        "step": 1.0,
                        "display": "slider",
                        "label": "SLIC Weight",
                    },
                ),
            },
        }
        return params

    @classmethod
    def INPUT_TYPES(cls):
        """
        定义节点的输入类型

        返回:
            dict: 输入参数的定义，包括图像和SLIC参数
        """
        params = cls.get_slic_params()
        return params

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "pixelate_image_node"
    CATEGORY = "ComfyUI-Seed-Nodes"

    def pixelate_image_node(self, image, step=10, iters=10, stride=10, weight=10):
        """
        执行基于 SLIC 的图像像素化处理。

        参数:
            image (torch.Tensor): 输入图像张量，形状应为 (C, H, W) 或 (1, C, H, W)。
            step (int): SLIC 网格步长。 
            iters (int): SLIC 迭代次数。
            stride (int): SLIC 像素化步幅。
            weight (float): SLIC 颜色距离权重。

        返回:
            torch.Tensor: 像素化后的图像张量。
        """
        # 确保输入为 torch.Tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)

        # 预处理输入图像
        input_image = image.cpu().numpy()
        input_image = self.preprocess_input(input_image)

        # 归一化到 [0, 255] 并转换为 uint8
        if input_image.max() <= 1.0:
            input_image = (input_image * 255).astype(np.uint8)
        else:
            input_image = input_image.astype(np.uint8)

        # 检查通道数
        if input_image.shape[-1] not in [3, 4]:
            raise ValueError(f"SLICPixelatorNode: 支持的通道数为 3 或 4，但收到 {input_image.shape[-1]}")

        # 如果有 alpha 通道，去除它用于 SLIC 处理
        has_alpha = input_image.shape[-1] == 4
        if has_alpha:
            image_np = input_image[:, :, :3]
            alpha_channel = input_image[:, :, 3]
        else:
            image_np = input_image

        # 初始化 SLIC 对象
        slic = SLIC(image_np, step=step, iters=iters, stride=stride, weight=weight)

        # 执行 SLIC 并获取像素化图像
        result = slic.pixel_deal()

        # 如果原图有 alpha 通道，保留 alpha
        if has_alpha:
            result = np.dstack((result, alpha_channel)).astype(np.uint8)
            img = Image.fromarray(result, 'RGBA')
        else:
            img = Image.fromarray(result, 'RGB')

        # 转换回张量
        result_tensor = self.convert_to_tensor(img, image.shape, image.device)

        return (result_tensor,)

    def preprocess_input(self, input_image):
        """
        预处理输入数组，处理特殊形状和通道顺序。

        参数:
            input_image (numpy.ndarray): 输入图像数组。

        返回:
            numpy.ndarray: 预处理后的图像数组。
        """
        # 如果输入有形状 (batch, C, H, W)，则移除批处理维度
        if len(input_image.shape) == 4:
            input_image = input_image[0]  # 形状: (C, H, W)

        # 如果输入有形状 (C, H, W)，则转置为 (H, W, C)
        if len(input_image.shape) == 3 and input_image.shape[0] in [1, 3, 4]:
            input_image = np.transpose(input_image, (1, 2, 0))  # 形状: (H, W, C)

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

        # 转换为 CxHxW
        if compressed_array.ndim == 3:
            compressed_array = np.transpose(compressed_array, (2, 0, 1))

        return torch.from_numpy(compressed_array).to(device)
