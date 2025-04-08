import json
import os
from datasets import Dataset, DatasetDict
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.9'

def load_conversations(data_path, file_paths):
    conversations = []
    for file_path in file_paths:
        file_path = os.path.join(data_path, file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            dialog = json.load(f)
            conversation = []
            for i in range(1, len(dialog)):
                if dialog[i]["role"] == "user": continue
                patient = dialog[i-1]["content"]
                doctor = dialog[i]["content"]
                conversation.append({
                    'patient': patient,
                    'doctor': doctor
                })
            conversations.extend(conversation)
    return conversations

import random
train_path = "./data/Chats/Output-1/"
test_path = "./data/SMILE-ChatDdata/"

train_file_paths = [f for f in os.listdir(train_path) if f.endswith(".json")]
test_file_paths = [f for f in os.listdir(train_path) if f.endswith(".json")]
select_train_file_paths = random.sample(train_file_paths, 10)
select_test_file_paths = random.sample(train_file_paths, 10)

train_data = load_conversations(train_path, select_train_file_paths)
test_data = load_conversations(train_path, select_test_file_paths)

train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)





from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

# 加载模型和tokenizer
print('--model--')
model_name = "/root/autodl-tmp/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# LoRA配置
lora_config = LoraConfig(
    r=8,  # LoRA的rank参数
    lora_alpha=16,  # LoRA的alpha参数
    lora_dropout=0.1,  # LoRA的dropout
    target_modules=["q_proj", "k_proj", "v_proj"],  # LoRA调整的模块
    # lora_scale=32  # LoRA缩放系数
)

# 使用LoRA微调模型
model = get_peft_model(model, lora_config)

# 准备训练数据
# def preprocess_function(examples):
#     # 将对话内容拼接起来以适应模型输入
#     input_texts = [f"下面是心理健康医生和病患的对话，你需要学习医生的共情对话能力。{item['patient']};{item['doctor']}" for item in examples['patient']]
#     return tokenizer(input_texts, truncation=True, padding="max_length", max_length=512)
def preprocess_function(examples):
    # 将对话内容拼接起来以适应模型输入
    input_texts = [f"下面是心理健康医生和病患的对话，你需要学习医生的共情对话能力。{patient} | {doctor}" for patient, doctor in zip(examples['patient'], examples['doctor'])]
    return tokenizer(input_texts, truncation=True, padding="max_length", max_length=512)

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)





from transformers import Trainer, TrainingArguments

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

# 创建Trainer对象
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# 开始训练
print('--train--')
trainer.train()




from evaluate import load
import nltk
from nltk.util import ngrams
from collections import Counter


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 计算 BLEU-1/2/3
    bleu = load("bleu")
    bleu_1 = bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels], n_val=1)["bleu"]
    bleu_2 = bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels], n_val=2)["bleu"]
    bleu_3 = bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels], n_val=3)["bleu"]

    # 计算 Distinct-1/2/3
    distinct_1 = distinct_n_gram(decoded_preds, n=1)
    distinct_2 = distinct_n_gram(decoded_preds, n=2)
    distinct_3 = distinct_n_gram(decoded_preds, n=3)

    return {
        "bleu-1": bleu_1,
        "bleu-2": bleu_2,
        "bleu-3": bleu_3,
        "distinct-1": distinct_1,
        "distinct-2": distinct_2,
        "distinct-3": distinct_3
    }

# 计算 Distinct-N 的方法
def distinct_n_gram(predictions, n):
    n_grams = []
    for pred in predictions:
        words = pred.split()  # 将预测文本拆分为单词
        n_grams.extend(ngrams(words, n))  # 生成n-gram
    n_gram_counts = Counter(n_grams)  # 计算n-gram的频率
    distinct_n = len(n_gram_counts) / float(sum(n_gram_counts.values()))  # Distinct-N 公式
    return distinct_n

print('--eval--')
import torch
torch.cuda.empty_cache()

# with torch.no_grad():
#     trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="eval")
