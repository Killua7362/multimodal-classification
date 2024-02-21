from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.functions import monotonically_increasing_id
import urllib
import os
import shutil
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration,LlavaProcessor,LlamaTokenizerFast,CLIPImageProcessor
from PIL import Image
from peft import get_peft_model,LoraConfig,prepare_model_for_kbit_training
import bitsandbytes as bnb
from transformers import Trainer,TrainingArguments

spark = SparkSession.builder.getOrCreate()

def download_image(id,url):
    path = os.path.join('dataset/images',str(id)+'.jpg')
    if os.path.exists(path):
        return path
    try:
        with urllib.request.urlopen(url,timeout=5) as urldata,open(path,'wb') as out_file:
            shutil.copyfileobj(urldata,out_file)
            return path
    except:
        return None

df = spark.read.csv('dataset/data.tsv',sep=r'\t',header=True).select('clean_title','image_url','hasImage','6_way_label').limit(1000)
df = df.filter((df.hasImage == 'True') & (df.hasImage != 'nan') & (df.hasImage != ''))
df= df.drop('hasImage')
df = df.withColumn('id',monotonically_increasing_id())
udf_function = udf(download_image,StringType())
df = df.withColumn('local_path',udf_function(df['id'],df['image_url']))
df = df.filter(df['local_path'].isNotNull())
df = df.drop('image_url')

epochs = 4
batches = 32
max_length = 100
device = torch.device('cuda')
model_name = "llava-hf/bakLlava-v1-hf"
n_labels = 6

class RedditDataset(Dataset):
    def __init__(self,df,use_tokenizer,max_sequence_length):
        #inputs
        images = []
        texts = []
        print("Reading the Partitions")
        for row in tqdm(df.collect()):
            cap = row['6_way_label']
            title = row['clean_title']
            local_path = row['local_path']
            img = Image.open(open(local_path,'rb'))
            images.append(img)
            texts.append(self.format_prompt(title,text_mapping[int(cap)]))
            
        self.n_examples = len(texts)
        self.inputs = use_tokenizer(texts,images,return_tensors='pt', truncation=True,padding=True,max_length=200)
        self.sequence_length = self.inputs['input_ids'][0].shape[-1]
    
    def format_prompt(self,caption,output=None):
        intro = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        intro_data = """Instruction:\nCategorize relation between the image and caption into one of the 6 categories:
                \n\nTrue NEWS\nSatire\nMisleading Content\nManipulated Content\nFalse Connection\nImposter Content"""
        _input = f"USER: <image>\n{caption}"
        dummy = ""
        _output = f"ASSISTANT:\n{output if output else dummy}"
        
        return "\n\n".join([intro,intro_data,_input,_output])
    
    def __len__(self):
        return self.n_examples
    
    def __getitem__(self,item):
        return {key:self.inputs[key][item] for key in self.inputs.keys()}

model = LlavaForConditionalGeneration.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    load_in_4bit=True,
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
text_mapping = ["True","Satire","Misleading Content","Manipulated Content","False Connection","Imposter Content"]
image_tokenizer = CLIPImageProcessor.from_pretrained(model_name)
text_tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
# text_tokenizer.add_special_tokens({"pad_token":"<pad>"})
tokenizer = LlavaProcessor(image_processor=image_tokenizer,tokenizer=text_tokenizer)

cls = bnb.nn.Linear4bit
lora_module_names = set()
for name,module in model.named_modules():
    if isinstance(module,cls):
        names = name.split('.')
        lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
        
lora_module_names = list(lora_module_names)

peft_config = LoraConfig(
    r = 16,
    lora_alpha = 64,
    target_modules = lora_module_names,
    lora_dropout = 0.1,
    bias = 'none',
    task_type = "CAUSAL_LM",
)
    
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
else:
    def make_inputs_require_grad(module, input, output):
         output.requires_grad_(True)

    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
model = get_peft_model(model,peft_config)


train_dataset = RedditDataset(
    df=df,
    use_tokenizer=tokenizer,
    max_sequence_length=max_length
)


trainer = Trainer(
    model=model,
    train_dataset = train_dataset,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 2,
        max_steps = 100,
        learning_rate = 2e-4,
        fp16 = True,
        logging_steps = 1,
        output_dir = './results',
        optim = "paged_adamw_32bit",
        num_train_epochs=50,
    ),
)

train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
output_dir = 'saved_models'
trainer.model.save_pretrained(output_dir)


#merged_model = model_to_merge.merge_and_unload()
#merged_model.save_pretrained('llora_model')