import os
import sys
import argparse
import gc
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer
import folder_paths
import torchvision.transforms.functional as TF
from .src.wan_vae import AutoencoderKLWan
#from .src.wan_image_encoder import  CLIPModel
#from .src.wan_text_encoder import  WanT5EncoderModel
from .src.wan_transformer3d_audio_2512 import WanTransformerAudioMask3DModel
from .src.pipeline_wan_fun_inpaint_audio_2512 import WanFunInpaintAudioPipeline

from .src.utils import (filter_kwargs, get_image_to_video_latent, get_image_to_video_latent2,
                                   save_videos_grid)

from .src.fm_solvers import FlowDPMSolverMultistepScheduler
from .src.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .src.cache_utils import get_teacache_coefficients
from .infer import encode_prompt,get_image_to_video_latent3
# import decord
import json
import random
import math
import comfy.model_management as mm
import librosa
try:
    from moviepy.editor import  VideoFileClip, AudioFileClip
except:
    try:
        from moviepy import VideoFileClip, AudioFileClip
    except:
        from moviepy import *
import pyloudnorm as pyln
from transformers import Wav2Vec2FeatureExtractor
from  .src.wav2vec2 import Wav2Vec2Model
from einops import rearrange

def clear_comfyui_cache():
    cf_models=mm.loaded_models()
    try:
        for pipe in cf_models:
            pipe.unpatch_model(device_to=torch.device("cpu"))
            print(f"Unpatching models.{pipe}")
    except: pass
    mm.soft_empty_cache()
    torch.cuda.empty_cache()
    max_gpu_memory = torch.cuda.max_memory_allocated()
    print(f"After Max GPU memory allocated: {max_gpu_memory / 1000 ** 3:.2f} GB")



def get_sample_size(pil_img, sample_size):
    w, h = pil_img.size
    ori_a = w * h
    default_a = sample_size[0] * sample_size[1]
    if default_a < ori_a:
        ratio_a = math.sqrt(ori_a / sample_size[0] / sample_size[1])

        w = w / ratio_a // 16 * 16
        h = h / ratio_a // 16 * 16
    else:
        w = w // 16 * 16
        h = h // 16 * 16

    return [int(h), int(w)]


def get_ip_mask(coords):
    y1, y2, x1, x2, h, w = coords
    Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    mask = (Y.unsqueeze(-1) >= y1) & (Y.unsqueeze(-1) < y2) & (X.unsqueeze(-1) >= x1) & (X.unsqueeze(-1) < x2)
    
    mask = mask.reshape(-1)
    return mask.float()

def get_audio_embed(mel_input, wav2vec_feature_extractor, audio_encoder, video_length, sr=16000, fps=25, device='cpu'):

    audio_feature = np.squeeze(wav2vec_feature_extractor(mel_input, sampling_rate=sr).input_values)
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
    audio_feature = audio_feature.unsqueeze(0)

    # audio encoder
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)
        #embeddings = audio_encoder(audio_feature, output_hidden_states=True)

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")

    audio_emb = audio_emb.cpu().detach()
    return audio_emb

def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio


def load_v3_flash(
    sampler_name,
    vae_path,
    inp_vae,
    weigths_current_path,
    config_path,
    node_dir,
    use_mmgp,
    device,
    fsdp_dit = True,
    weight_dtype_str = "bfloat16",
    block_offload = False
):
    weight_dtype = torch.bfloat16 if weight_dtype_str == "bfloat16" else torch.float16
    wav2vec_model_dir=os.path.join(weigths_current_path,"chinese-wav2vec2-base")
    # # Load audio models
    global audio_encoder
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_model_dir, local_files_only=True).to('cuda')
    audio_encoder.feature_extractor._freeze_parameters()
    global wav2vec_feature_extractor
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model_dir, local_files_only=True)

    config = OmegaConf.load(config_path)
    model_name = os.path.join(node_dir,"Wan2.1-Fun-V1.1-1.3B-InP")
    transformer_path = os.path.join(weigths_current_path,"echomimicv3-flash-pro")
    transformer = WanTransformerAudioMask3DModel.from_pretrained(
        transformer_path,
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True if not fsdp_dit else False,
        torch_dtype=weight_dtype,
    )

    # Get Vae
    vae_path_=folder_paths.get_full_path("vae", vae_path)
    inp_vae_path=folder_paths.get_full_path("vae", inp_vae) if inp_vae != "none" else None
    vae = AutoencoderKLWan.from_pretrained(vae_path_,
        #os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)

    if inp_vae_path is not None:
        print(f"From checkpoint: {inp_vae_path}")
        if inp_vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(inp_vae_path)
        else:
            state_dict = torch.load(inp_vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    # Get Scheduler
    Choosen_Scheduler = scheduler_dict = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[sampler_name]
    if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
        config['scheduler_kwargs']['shift'] = 1
    scheduler = Choosen_Scheduler(
        **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Get Pipeline
    pipeline = WanFunInpaintAudioPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        #text_encoder=text_encoder,
        scheduler=scheduler,
        #clip_image_encoder=clip_image_encoder
    )


    if use_mmgp!="None":
        from mmgp import offload, profile_type
        pipeline.to("cpu")
        if use_mmgp=="VerylowRAM_LowVRAM":
            offload.profile(pipeline, profile_type.VerylowRAM_LowVRAM,quantizeTransformer=config.quantize_transformer)
        elif use_mmgp=="LowRAM_LowVRAM":  
            offload.profile(pipeline, profile_type.LowRAM_LowVRAM,quantizeTransformer=config.quantize_transformer)
        elif use_mmgp=="LowRAM_HighVRAM":
            offload.profile(pipeline, profile_type.LowRAM_HighVRAM,quantizeTransformer=config.quantize_transformer)
        elif use_mmgp=="HighRAM_LowVRAM":
            offload.profile(pipeline, profile_type.HighRAM_LowVRAM,quantizeTransformer=config.quantize_transformer)
        elif use_mmgp=="HighRAM_HighVRAM":
            offload.profile(pipeline, profile_type.HighRAM_HighVRAM,quantizeTransformer=config.quantize_transformer)
    elif block_offload:
        pipeline.to("cpu")
    else:
        pipeline.to(device)
    temporal_compression_ratio=pipeline.vae.config.temporal_compression_ratio
    return pipeline,temporal_compression_ratio,tokenizer


compiled = False

def infer_flash(pipeline,audio_embeds,prompt_embeds,negative_prompt_embeds,clip_context,fps,num_inference_steps,seed,video_length_actual,device,block_offload,
                input_video,input_video_mask,sample_height,sample_width,guidance_scale,latent_frames,audio_file_prefix,enable_riflex=False,
                enable_teacache=False,teacache_offload=False,teacache_threshold=0.1,riflex_k=6,audio_scale=1.0,use_un_ip_mask=False,
                num_skip_start_steps=5,audio_guidance_scale=3.0,neg_scale=1.0,neg_steps=0,use_dynamic_cfg=False,
                use_dynamic_acfg=False,cfg_skip_ratio=0.0,shift=5.0,
                ):
    coefficients = get_teacache_coefficients("Wan2.1-Fun-V1.1-1.3B-InP") if enable_teacache else None
    if coefficients is not None:
        print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
        pipeline.transformer.enable_teacache(
            coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
        )

    generator = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        if enable_riflex:
            pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)

        sample = pipeline(
            None, 
            num_frames = video_length_actual,
            negative_prompt = None,
            audio_embeds = audio_embeds,
            audio_scale=audio_scale,
            ip_mask = None,
            use_un_ip_mask=use_un_ip_mask,
            height      = sample_height,
            width       = sample_width,
            generator   = generator,
            neg_scale = neg_scale,
            neg_steps = neg_steps,
            use_dynamic_cfg=use_dynamic_cfg,
            use_dynamic_acfg=use_dynamic_acfg,
            guidance_scale = guidance_scale,
            audio_guidance_scale = audio_guidance_scale,
            num_inference_steps = num_inference_steps,
            video      = input_video,
            mask_video   = input_video_mask,
            clip_image = None,
            cfg_skip_ratio = cfg_skip_ratio,
            shift = shift,
            clip_context = clip_context,
            prompt_embeds = prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            block_offload=block_offload,
        ).videos

        # Save temporary video
        tmp_video_path = os.path.join(folder_paths.output_directory, f"{audio_file_prefix}_tmp.mp4")
        pli_list=save_videos_grid(sample[:,:,:video_length_actual], tmp_video_path, fps=fps)
        global compiled
        if not compiled:
            pipeline.transformer.compile()
            compiled = True
    return pli_list

def Flash_Echo_v3_predata(
    clip_image_encoder,
    text_encoder,
    tokenizer,
    prompt,
    negative_prompt,
    ref_image,
    sample_size,
    audio_path,
    weigths_current_path,
    fps,
    video_length,
    temporal_compression_ratio,
    device,
    weight_dtype
):
    wav2vec_model_dir=os.path.join(weigths_current_path,"chinese-wav2vec2-base")
    # Load audio
    #audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_model_dir, local_files_only=True).to('cpu')
    #audio_encoder.feature_extractor._freeze_parameters()
    #wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model_dir, local_files_only=True)
   
    audio_clip = AudioFileClip(audio_path)
    video_length_actual = min(int(audio_clip.duration * fps), video_length)
    num = (video_length_actual - 1) // temporal_compression_ratio * temporal_compression_ratio
    video_length_actual = int(num) + 1 if video_length_actual != 1 else 1

    # Get audio features
    mel_input, sr = librosa.load(audio_path, sr=16000) #TODO: 
    mel_input = loudness_norm(mel_input, sr)
    mel_input = mel_input[:int(video_length_actual / 25 * sr)]
    
    print(f"Audio length: {int(len(mel_input)/ sr * 25)}, Video length: {video_length_actual}")
    audio_feature_wav2vec = get_audio_embed(mel_input, wav2vec_feature_extractor, audio_encoder, video_length_actual, sr=16000, fps=25, device='cuda')

    # Get audio batch 
    audio_embeds = audio_feature_wav2vec.to(device=device, dtype=weight_dtype)
    
    indices = (torch.arange(2 * 2 + 1) - 2) * 1 
    center_indices = torch.arange(
        0,  
        video_length_actual,
        1,).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=audio_embeds.shape[0]-1)
    audio_embeds = audio_embeds[center_indices] # F w s c [F, 5, 12, 768]
    audio_embeds = audio_embeds.unsqueeze(0).to(device=device)
    print(f"Audio embeds shape: {audio_embeds.shape}") #Audio embeds shape: torch.Size([1, 97, 5, 12, 768])

    # Load reference image
    #ref_image = Image.open(image_path).convert("RGB")

    ref_start = np.array(ref_image)
    validation_image_start = Image.fromarray(ref_start).convert("RGB")

    validation_image_end = None
    latent_frames = (video_length_actual - 1) // temporal_compression_ratio + 1
    sample_height, sample_width = get_sample_size(validation_image_start, sample_size)


    input_video, input_video_mask, clip_image = get_image_to_video_latent2(validation_image_start, 
                                                                           validation_image_end, video_length=video_length_actual, sample_size=[sample_height, sample_width])
    # get clip image

    # video_length = init_frames + partial_video_length

    if clip_image is not None:
        clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(device) #torch.Size([3, 1008, 576])
       
        #clip_context = self.clip_image_encoder([clip_image[:, None, :, :]])
        clip_image=clip_image.permute(1, 2, 0).unsqueeze(0) #comfy need [B,C,H,W]
        clip_dict=clip_image_encoder.encode_image(clip_image)
        clip_context =clip_dict["penultimate_hidden_states"].to(device, weight_dtype)
        #print(clip_dict["image_embeds"].shape,clip_dict["last_hidden_state"].shape,clip_dict["penultimate_hidden_states"].shape,) #torch.Size([1, 1024]) torch.Size([1, 257, 1280]) torch.Size([1, 257, 1280])
       
        #print(clip_context.shape) #torch.Size([1, 257, 1280])
    else:
        clip_image = Image.new("RGB", (512, 512), color=(0, 0, 0))  
        clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(device) 
        #clip_context = self.clip_image_encoder([clip_image[:, None, :, :]])
        clip_image=clip_image.permute(1, 2, 0).unsqueeze(0)
        clip_context =clip_image_encoder.encode_image(clip_image)["penultimate_hidden_states"].to(device, weight_dtype)
        clip_context = torch.zeros_like(clip_context)
    clip_context=clip_context.to(device=device, dtype=weight_dtype)
    clear_comfyui_cache()
    gc.collect()

    prompt_embeds, negative_prompt_embeds=encode_prompt(text_encoder,tokenizer,prompt,negative_prompt,True,1,device=device,dtype=weight_dtype)
    clear_comfyui_cache()
    

    emb={
        "audio_embeds":audio_embeds,"video_length":video_length,"clip_context":clip_context,"sample_height":sample_height,"sample_width":sample_width,
         "video_length_actual":video_length_actual,"input_video":input_video,"input_video_mask":input_video_mask,
         "prompt_embeds":prompt_embeds,"negative_prompt_embeds":negative_prompt_embeds,
          "ref_image_pil":ref_image,"latent_frames":latent_frames,
    }
    return emb


