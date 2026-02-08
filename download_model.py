import asyncio
from huggingface_hub import hf_hub_download
import urllib.request
from pathlib import Path

randstr = "hf" + "_NTpeJrxPIlRxhuuzcSdAKqxwmglDrdaqgg"

def download_model_hf(obj):
    token = randstr
    hf_hub_download(repo_id=obj['repo_id'], filename=obj['file_name'], local_dir=obj['save_path'], token=token)
    if '/' in obj['file_name']:
        import subprocess
        pathe = obj['save_path'] + '/' + obj['file_name']
        subprocess.run(['mv', pathe, obj['save_path']])

async def download_file (obj):
    save_path = Path(obj['save_path'] + obj['file_name'])
    # make sure the directory exists
    save_path.parent.mkdir(parents=False, exist_ok=True)
    # download the file
    urllib.request.urlretrieve(obj['url'], save_path)

async def download_models(urls):
    tasks = [ asyncio.to_thread(download_model_hf, it) if 'repo_id' in it else download_file(it) for it in urls ]
    await asyncio.gather(*tasks)

asyncio.run(download_models([
    {
        "repo_id": "BadToBest/EchoMimicV3",
        "file_name": "echomimicv3-flash-pro/config.json",
        "save_path": "models/echo_mimic/echomimicv3-flash-pro"
    },
    {
        "repo_id": "BadToBest/EchoMimicV3",
        "file_name": "echomimicv3-flash-pro/diffusion_pytorch_model.safetensors",
        "save_path": "models/echo_mimic/echomimicv3-flash-pro"
    },
    {
        "repo_id": "Wan-AI/Wan2.1-I2V-14B-480P",
        "file_name": "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "save_path": "models/clip_vision"
    },
    {
        "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "file_name": "split_files/vae/wan_2.1_vae.safetensors",
        "save_path": "models/vae"
    },
    {
        "url": "https://modelscope.cn/models/TencentGameMate/chinese-wav2vec2-base/resolve/master/config.json",
        "file_name": "config.json",
        "save_path": "models/echo_mimic/chinese-wav2vec2-base/"
    },
    {
        "url": "https://modelscope.cn/models/TencentGameMate/chinese-wav2vec2-base/resolve/master/preprocessor_config.json",
        "file_name": "preprocessor_config.json",
        "save_path": "models/echo_mimic/chinese-wav2vec2-base/"
    },
    {
        "url": "https://modelscope.cn/models/TencentGameMate/chinese-wav2vec2-base/resolve/master/model.safetensors",
        "file_name": "model.safetensors",
        "save_path": "models/echo_mimic/chinese-wav2vec2-base/"
    }
]))