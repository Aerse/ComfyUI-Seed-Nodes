import os
import io
import json
import torch
import torchaudio
import folder_paths
import struct
from comfy.cli_args import args
import glob

def create_vorbis_comment_block(comment_dict, last_block):
    vendor_string = b'ComfyUI'
    vendor_length = len(vendor_string)

    comments = []
    for key, value in comment_dict.items():
        comment = f"{key}={value}".encode('utf-8')
        comments.append(struct.pack('<I', len(comment)) + comment)

    user_comment_list_length = len(comments)
    user_comments = b''.join(comments)

    comment_data = struct.pack('<I', vendor_length) + vendor_string + struct.pack('<I', user_comment_list_length) + user_comments
    if last_block:
        id = b'\x84'
    else:
        id = b'\x04'
    comment_block = id + struct.pack('>I', len(comment_data))[1:] + comment_data

    return comment_block

def insert_or_replace_vorbis_comment(flac_io, comment_dict):
    if len(comment_dict) == 0:
        return flac_io

    flac_io.seek(4)

    blocks = []
    last_block = False

    while not last_block:
        header = flac_io.read(4)
        last_block = (header[0] & 0x80) != 0
        block_type = header[0] & 0x7F
        block_length = struct.unpack('>I', b'\x00' + header[1:])[0]
        block_data = flac_io.read(block_length)

        if block_type == 4 or block_type == 1:
            pass
        else:
            header = bytes([(header[0] & (~0x80))]) + header[1:]
            blocks.append(header + block_data)

    blocks.append(create_vorbis_comment_block(comment_dict, last_block=True))

    new_flac_io = io.BytesIO()
    new_flac_io.write(b'fLaC')
    for block in blocks:
        new_flac_io.write(block)

    new_flac_io.write(flac_io.read())
    return new_flac_io

class SeedSaveAudio:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                "audio": ("AUDIO", ),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "format": (["flac", "wav", "mp3"], {"default": "flac"})
                },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_audio"

    OUTPUT_NODE = True

    CATEGORY = "audio"

    def save_audio(self, audio, filename_prefix="ComfyUI", format="flac", prompt=None, extra_pnginfo=None):
        # 确保文件名前缀不包含路径分隔符
        filename_prefix = os.path.basename(filename_prefix)
        filename_prefix += self.prefix_append
        
        # 在output/audio目录下保存
        audio_output_dir = os.path.join(self.output_dir, "audio")
        os.makedirs(audio_output_dir, exist_ok=True)
        
        # 获取保存路径信息
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, audio_output_dir)
        results = list()

        metadata = {}
        if not args.disable_metadata:
            if prompt is not None:
                metadata["prompt"] = json.dumps(prompt)
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])

        # 获取当前目录下最新的计数器
        pattern = os.path.join(full_output_folder, f"{filename_prefix}_*.{format}")
        existing_files = glob.glob(pattern)
        if existing_files:
            # 从现有文件名中提取最大计数器值
            numbers = [int(f.split('_')[-1].split('.')[0]) for f in existing_files]
            counter = max(numbers) + 1
        else:
            counter = 0

        for (batch_number, waveform) in enumerate(audio["waveform"].cpu()):
            # 使用完整的文件名前缀
            file = f"{filename_prefix}_{counter:05}.{format}"
            output_path = os.path.join(full_output_folder, file)

            if format == "flac":
                buff = io.BytesIO()
                torchaudio.save(buff, waveform, audio["sample_rate"], format="FLAC")
                # 只有FLAC格式支持元数据
                buff = insert_or_replace_vorbis_comment(buff, metadata)
                with open(output_path, 'wb') as f:
                    f.write(buff.getbuffer())
            elif format == "mp3":
                # MP3格式需要特殊处理
                # 确保音频是正确的格式和范围
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                # 标准化音频到 [-1, 1] 范围
                waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
                # MP3格式直接保存
                torchaudio.save(output_path, waveform, audio["sample_rate"], format="MP3")
            else:  # wav格式
                # WAV格式标准化处理
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
                torchaudio.save(output_path, waveform, audio["sample_rate"], format="WAV")

            results.append({
                "filename": file,
                "subfolder": "audio",
                "type": self.type
            })
            counter += 1

        return { "ui": { "audio": results } } 