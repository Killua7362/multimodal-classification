from trl import DPOTrainer
from PIL import Image
from utils.data import format_prompt
import torch

class CustomDPOTrainer(DPOTrainer):
    text_tokenizer:None
    
    def tokenize_data(self,feature,model):
        text_mapping = ["True","Satire","Misleading Content","Manipulated Content","False Connection","Imposter Content"]
        clean_title = feature['clean_title']
        answer = feature['6_way_label']
        image_path = feature['local_path']
        chosen = []
        rejected = []
        prompt = []
        img_prompt = []
        batch = {}
        for i in range(len(clean_title)):
            img = Image.open(image_path[i])
            idx = int(answer[i])
            #choosen
            out = text_mapping[idx]
            #rejected
            for i in range(len(text_mapping)):
                if i != idx:
                    chosen.append(out)
                    rejected.append(text_mapping[i])
                    img_prompt.append(img)
                    prompt.append(format_prompt(clean_title[i]))
        
        chosen_tokens = self.text_tokenizer(
            chosen, truncation=True, max_length=self.max_target_length,padding=True, add_special_tokens=False
        )
        rejected_tokens = self.text_tokenizer(
            rejected, truncation=True, max_length=self.max_target_length,padding=True, add_special_tokens=False
        )
        prompt_tokens = self.tokenizer(
            img_prompt,prompt, truncation=True, max_length=self.max_prompt_length,padding=True, add_special_tokens=False
        )

        batch["chosen_labels"] = chosen_tokens["input_ids"]
        batch["rejected_labels"] = rejected_tokens["input_ids"]
        batch["prompt_input_ids"] = prompt_tokens["input_ids"]
        batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]
        batch['pixel_values'] = prompt_tokens["pixel_values"]
        
        if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
            batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                labels=torch.tensor(batch["rejected_labels"])
            )
            batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                labels=torch.tensor(batch["chosen_labels"])
            )

        return batch
