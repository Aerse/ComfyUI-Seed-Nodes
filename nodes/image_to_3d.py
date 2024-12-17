import os
import base64
import requests
import json
import torch
import numpy as np
from PIL import Image
import io
import folder_paths


class ImageTo3DNode:
    """
    将图像转换为3D模型的节点,并保存GLB文件到输出目录
    """
    
    def __init__(self):
        self.api_endpoint = "http://localhost:15978/generate"
        self.output_dir = folder_paths.get_output_directory()
        self.type = "glb"
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        定义节点的输入参数
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "randomize_seed": ("BOOLEAN", {
                    "default": True,
                }),
                "ss_guidance_strength": ("FLOAT", {
                    "default": 7.5,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                }),
                "ss_sampling_steps": ("INT", {
                    "default": 12,
                    "min": 1,
                    "max": 50,
                }),
                "slat_guidance_strength": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                }),
                "slat_sampling_steps": ("INT", {
                    "default": 12,
                    "min": 1,
                    "max": 50,
                }),
                "mesh_simplify": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.9,
                    "max": 0.98,
                    "step": 0.01,
                }),
                "texture_size": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 512,
                })
            }
        }

    RETURN_TYPES = ("STRING",)  # 返回保存的GLB文件路径
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "generate_3d"
    CATEGORY = "ComfyUI-Seed-Nodes"

    def download_glb(self, glb_url):
        """
        从API响应的URL下载GLB文件并保存到输出目录
        """
        try:
            # 构建完整的URL
            full_url = f"http://localhost:15978{glb_url}"
            
            # 下载文件
            response = requests.get(full_url)
            response.raise_for_status()
            
            # 从URL中提取文件名
            filename = os.path.basename(glb_url)
            # 添加model_前缀
            filename = f"model_{filename}"
            
            # 构建保存路径
            save_path = os.path.join(self.output_dir, filename)
            
            # 保存文件
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            return save_path
            
        except Exception as e:
            print(f"Error downloading GLB file: {str(e)}")
            return None

    def image_to_base64(self, image_tensor):
        """
        将图像张量转换为base64字符串
        """
        # 确保输入为torch.Tensor
        if not isinstance(image_tensor, torch.Tensor):
            image_tensor = torch.from_numpy(image_tensor)
        
        # 转换为PIL图像
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]
        
        # 转换通道顺序从(C,H,W)到(H,W,C)
        image_np = image_tensor.cpu().numpy()
        if image_np.shape[0] in [3, 4]:
            image_np = np.transpose(image_np, (1, 2, 0))
        
        # 确保值范围在0-255之间
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        
        # 转换为PIL图像
        if image_np.shape[-1] == 4:
            img = Image.fromarray(image_np, 'RGBA')
        else:
            img = Image.fromarray(image_np, 'RGB')
        
        # 转换为base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def generate_3d(self, image, seed=0, randomize_seed=True,
                   ss_guidance_strength=7.5, ss_sampling_steps=12,
                   slat_guidance_strength=3.0, slat_sampling_steps=12,
                   mesh_simplify=0.95, texture_size=1024):
        """
        生成3D模型并保存到文件
        """
        try:
            # 准备请求数据
            base64_image = self.image_to_base64(image)
            
            data = {
                "images": [base64_image],
                "seed": seed,
                "randomize_seed": randomize_seed,
                "ss_guidance_strength": ss_guidance_strength,
                "ss_sampling_steps": ss_sampling_steps,
                "slat_guidance_strength": slat_guidance_strength,
                "slat_sampling_steps": slat_sampling_steps,
                "mesh_simplify": mesh_simplify,
                "texture_size": texture_size
            }

            # 发送API请求
            response = requests.post(
                self.api_endpoint,
                headers={'Content-Type': 'application/json'},
                data=json.dumps(data)
            )
            
            # 检查响应
            if response.status_code == 200:
                result = response.json()
                if result.get('results'):
                    # 获取GLB URL并下载文件
                    glb_url = result['results'][0]['glb_url']
                    saved_path = self.download_glb(glb_url)
                    if saved_path:
                        return (saved_path,)
            
            raise Exception(f"API request failed: {response.text}")
            
        except Exception as e:
            print(f"Error generating 3D model: {str(e)}")
            return ("",)  # 返回空字符串表示失败

    def validate_inputs(self, **kwargs):
        """
        验证输入参数
        """
        valid_ranges = {
            "ss_guidance_strength": (0, 10),
            "ss_sampling_steps": (1, 50),
            "slat_guidance_strength": (0, 10),
            "slat_sampling_steps": (1, 50),
            "mesh_simplify": (0.9, 0.98),
            "texture_size": (512, 2048)
        }
        
        for param, (min_val, max_val) in valid_ranges.items():
            if param in kwargs:
                value = kwargs[param]
                if not min_val <= value <= max_val:
                    return False, f"{param} must be between {min_val} and {max_val}"
        
        return True, ""