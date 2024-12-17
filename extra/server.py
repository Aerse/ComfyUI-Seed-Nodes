from flask import Flask, request, jsonify,send_from_directory
from flask_cors import CORS
import base64
import io
from PIL import Image
import torch
import numpy as np
import uuid
import os
from typing import *
from easydict import EasyDict as edict
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils

app = Flask(__name__)
CORS(app)

# 配置
MAX_SEED = np.iinfo(np.int32).max
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 创建必要的目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 初始化pipeline
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

def preprocess_image(base64_string: str) -> Tuple[str, Image.Image]:
    """处理Base64图片数据"""
    try:
        # 解码Base64数据
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        # 生成唯一ID并处理图片
        trial_id = str(uuid.uuid4())
        processed_image = pipeline.preprocess_image(image)
        
        # 保存处理后的图片
        save_path = os.path.join(UPLOAD_FOLDER, f"{trial_id}.png")
        processed_image.save(save_path)
        
        return trial_id, processed_image
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")

def pack_state(gs: Gaussian, mesh: MeshExtractResult, trial_id: str) -> dict:
    """打包3D模型状态"""
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
        'trial_id': trial_id,
    }

def unpack_state(state: dict) -> Tuple[Gaussian, edict, str]:
    """解包3D模型状态"""
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')
    
    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )
    
    return gs, mesh, state['trial_id']

@app.route('/generate', methods=['POST'])
def generate():
    """处理生成3D模型的请求"""
    try:
        data = request.json
        if not data or 'images' not in data:
            return jsonify({'error': 'No images provided'}), 400

        results = []
        for image_data in data['images']:
            # 获取参数
            seed = data.get('seed', 0)
            randomize_seed = data.get('randomize_seed', True)
            ss_guidance_strength = data.get('ss_guidance_strength', 7.5)
            ss_sampling_steps = data.get('ss_sampling_steps', 12)
            slat_guidance_strength = data.get('slat_guidance_strength', 3.0)
            slat_sampling_steps = data.get('slat_sampling_steps', 12)
            mesh_simplify = data.get('mesh_simplify', 0.95)
            texture_size = data.get('texture_size', 1024)

            # 处理图片
            trial_id, processed_image = preprocess_image(image_data)

            # 如果需要随机种子
            if randomize_seed:
                seed = np.random.randint(0, MAX_SEED)

            # 生成3D模型
            outputs = pipeline.run(
                processed_image,
                seed=seed,
                formats=["gaussian", "mesh"],
                preprocess_image=False,
                sparse_structure_sampler_params={
                    "steps": ss_sampling_steps,
                    "cfg_strength": ss_guidance_strength,
                },
                slat_sampler_params={
                    "steps": slat_sampling_steps,
                    "cfg_strength": slat_guidance_strength,
                },
            )

            # 生成GLB文件
            state = pack_state(outputs['gaussian'][0], outputs['mesh'][0], trial_id)
            gs, mesh, _ = unpack_state(state)
            glb = postprocessing_utils.to_glb(
                gs, 
                mesh, 
                simplify=mesh_simplify, 
                texture_size=texture_size, 
                verbose=False
            )
            
            # 保存GLB文件
            glb_filename = f"{trial_id}.glb"
            glb_path = os.path.join(OUTPUT_FOLDER, glb_filename)
            glb.export(glb_path)

            # 构建结果URL
            glb_url = f"/download/{glb_filename}"
            results.append({
                'trial_id': trial_id,
                'glb_url': glb_url
            })

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """处理文件下载请求"""
    try:
        return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=15978)