from .nodes.load_image import LoadImage
from .nodes.image_pixelator import ImagePixelatorNode
from .nodes.load_multiple_images import LoadMultipleImages
from .nodes.image_slic_pixelator import SLICPixelatorNode

# 映射节点类到其在 ComfyUI 中的显示名称
NODE_CLASS_MAPPINGS = {
    "Seed-Nodes: LoadImage": LoadImage,
    "Seed-Nodes: ImagePixelator": ImagePixelatorNode,
    "Seed-Nodes: LoadMultipleImages": LoadMultipleImages,
    "Seed-Nodes: SLICPixelator": SLICPixelatorNode,
}

# 定义节点在 ComfyUI 中的显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "Seed-Nodes: LoadImage": "Seed-Nodes: LoadImage",
    "Seed-Nodes: ImagePixelator": "Seed-Nodes: ImagePixelator",
    "Seed-Nodes: LoadMultipleImages": "Seed-Nodes: LoadMultipleImages",
    "Seed-Nodes: SLICPixelator": "Seed-Nodes: SLICPixelator",
}