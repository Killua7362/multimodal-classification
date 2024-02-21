from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.functions import monotonically_increasing_id
import urllib
import os
import shutil
import torch
from torch.utils.data import Dataset
from transformers import LlavaForConditionalGeneration,LlavaProcessor,LlamaTokenizerFast,CLIPImageProcessor,TrainingArguments
from peft import prepare_model_for_kbit_training,PeftModel
from datasets import Dataset
from utils.dfo import CustomDPOTrainer

model_name = "llava-hf/bakLlava-v1-hf"
output_dir = 'saved_models'

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
df = Dataset.from_spark(df)

#reload the model
model = LlavaForConditionalGeneration.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    load_in_4bit=True,
)

image_tokenizer = CLIPImageProcessor.from_pretrained(model_name)
text_tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
tokenizer = LlavaProcessor(image_processor=image_tokenizer,tokenizer=text_tokenizer)

tokenizer.pad_token_id = model.config.eos_token_id
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)


model = PeftModel.from_pretrained(model,output_dir,is_trainable=True,adapter_name='Train')
model.load_adapter(output_dir,adapter_name='reference')

training_args = TrainingArguments(output_dir="./results")

train_dataset = {}

dpo_trainer = CustomDPOTrainer(
    model,
    args=training_args,
    beta=0.1,
    train_dataset=df,
    tokenizer=tokenizer,
    model_adapter_name="Train",
    ref_adapter_name="reference",
    remove_unused_columns=False,
    text_tokenizer = text_tokenizer
)

dpo_trainer.train()
dpo_trainer.save_model()

