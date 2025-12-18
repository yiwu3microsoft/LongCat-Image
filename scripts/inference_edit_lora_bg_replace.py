import torch
from PIL import Image
from transformers import AutoProcessor
from diffusers import LongCatImageTransformer2DModel
from diffusers import LongCatImageEditPipeline
from peft import PeftModel

if __name__ == '__main__':

    device = torch.device('cuda')
    checkpoint_dir = './weights/LongCat-Image-Edit'
    lora_ckpt = 'output/edit_lora_model/checkpoints-11000/transformer'

    text_processor = AutoProcessor.from_pretrained( checkpoint_dir, subfolder = 'tokenizer'  )
    transformer = LongCatImageTransformer2DModel.from_pretrained( checkpoint_dir , subfolder = 'transformer', 
        torch_dtype=torch.bfloat16, use_safetensors=True).to(device)

    print(f"Loading LoRA from {lora_ckpt} using PEFT...")
    transformer = PeftModel.from_pretrained(transformer, lora_ckpt, is_trainable=False)
    transformer = transformer.merge_and_unload()
    transformer = transformer.to(device, dtype=torch.bfloat16)
    transformer.eval()

    pipe = LongCatImageEditPipeline.from_pretrained(
        checkpoint_dir,
        transformer=transformer,
        text_processor=text_processor,
        torch_dtype=torch.bfloat16,
    )

    # pipe.to(device, torch.bfloat16)  # Uncomment for high VRAM devices (Faster inference)
    pipe.enable_model_cpu_offload()  # Offload to CPU to save VRAM (Required ~19 GB); slower but prevents OOM

    generator = torch.Generator("cpu").manual_seed(43)

    img_id = "a10e5e6f-a049-9695-b166-b73938f205fd"
    img = Image.open(f'assets/bg_replace/{img_id}.png').convert('RGB')
    # prompt = 'A bottle with capsules is placed on a table, with Vision & Blue Light text next to it, creating an informative atmosphere.'
    prompt = 'Put the product in the more relevant background or context and make it more recognizable and appealing.'
    image = pipe(
        img,
        prompt,
        negative_prompt='',
        guidance_scale=4.5,
        num_inference_steps=50,
        num_images_per_prompt=1,
        generator=generator
    ).images[0]

    image.save(f'/tmp/output/6fb1b1d1-4842-484f-8c08-b09a3199d1f8_3c53764c/results/edit_lora_11000_general-recontext_{img_id}.png')
