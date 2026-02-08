from infer_flash_pro import load_v3_flash, Flash_Echo_v3_predata, infer_flash



pipeline, clip_image_encoder, temporal_compression_ratio = load_v3_flash()


def execute(image_path: str, audio_path: str):


    emb = Flash_Echo_v3_predata(image_path, audio_path, clip_image_encoder, temporal_compression_ratio)
    infer_flash(
        pipeline,
        emb["audio_embeds"],
        emb["prompt_embeds"],
        emb["negative_prompt_embeds"],
        emb["clip_context"],
        42,
        emb["video_length_actual"],
        emb["input_video"],
        emb["input_video_mask"],
        emb["sample_height"],
        emb["sample_width"],
        emb["latent_frames"],
        "output"
    )