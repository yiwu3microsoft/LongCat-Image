import torch
from PIL import Image
from transformers import AutoProcessor
from longcat_image.models import LongCatImageTransformer2DModel
from longcat_image.pipelines import LongCatImageEditPipeline

if __name__ == '__main__':

    device = torch.device('cuda')
    checkpoint_dir = './weights/LongCat-Image-Edit'
    lora_dir = 'output/edit_lora_model/checkpoints-6000/transformer'
    text_processor = AutoProcessor.from_pretrained( checkpoint_dir, subfolder = 'tokenizer'  )
    transformer = LongCatImageTransformer2DModel.from_pretrained( checkpoint_dir , subfolder = 'transformer', 
        torch_dtype=torch.bfloat16, use_safetensors=True).to(device)

    pipe = LongCatImageEditPipeline.from_pretrained(
        checkpoint_dir,
        transformer=transformer,
        text_processor=text_processor,
        torch_dtype=torch.bfloat16,
    )

    
    # Load LoRA weights and (optionally) fuse
    # Common params: weight_name like 'pytorch_lora_weights.safetensors', and lora_scale.
    pipe.load_lora_weights(
        lora_dir,
        weight_name='adapter_model.safetensors',  # uncomment if your file name is fixed
        lora_scale=1.0
    )

    # (Optional) Fuse LoRA for slight speed-up and lower VRAM; after fusing, scale changes require reloading.
    # pipe.fuse_lora()  # uncomment if your pipeline supports fusing

    # pipe.to(device, torch.bfloat16)  # Uncomment for high VRAM devices (Faster inference)
    pipe.enable_model_cpu_offload()  # Offload to CPU to save VRAM (Required ~19 GB); slower but prevents OOM

    generator = torch.Generator("cpu").manual_seed(43)
    # ./data/data_background_change_11404/original/backgroud_change_training/art_collecting_data_30k_3category_ori_seg/58fa3da834a340ea0fa715933ab12c0f.jpg
    img = Image.open('./data/data_background_change_11404/masked_images/58fa3da834a340ea0fa715933ab12c0f.png').convert('RGB')
    prompt = 'Hat and casual outfit is placed on a scenic hilltop, with twisted branches surrounding it, and a serene landscape positioned behind it, evoking a tranquil atmosphere.'
    image = pipe(
        img,
        prompt,
        negative_prompt='',
        guidance_scale=4.5,
        num_inference_steps=50,
        num_images_per_prompt=1,
        generator=generator
    ).images[0]

    image.save('./output/edit_example_lora_6000.png')
