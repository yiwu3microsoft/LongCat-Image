import torch
from PIL import Image
from transformers import AutoProcessor
from longcat_image.models import LongCatImageTransformer2DModel
from longcat_image.pipelines import LongCatImageEditPipeline
from peft import PeftModel

if __name__ == '__main__':

    device = torch.device('cuda')
    checkpoint_dir = './weights/LongCat-Image-Edit'
    lora_ckpt = 'output/edit_lora_model/checkpoints-6000/transformer'

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
