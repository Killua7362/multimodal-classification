from PIL import Image
import torch
from transformers import LlavaForConditionalGeneration,LlavaProcessor,LlamaTokenizerFast,CLIPImageProcessor

model_name = "llava-hf/bakLlava-v1-hf"

image_tokenizer = CLIPImageProcessor.from_pretrained(model_name)
text_tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
tokenizer = LlavaProcessor(image_processor=image_tokenizer,tokenizer=text_tokenizer)


model = LlavaForConditionalGeneration.from_pretrained(
    "path to model", 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
)
input_text = ""
input_image_path = ""
input_image = Image.open(input_image_path)

inputs = tokenizer(text=input_text, images=input_image, return_tensors="pt")
generate_ids = model.generate(**inputs, max_length=30)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])