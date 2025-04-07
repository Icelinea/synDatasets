import json
import os
from datasets import Dataset, DatasetDict


def load_conversations(data_path, file_paths):
    conversations = []
    for file_path in file_paths:
        file_path = os.path.join(data_path, file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            dialog = json.load(f)
            conversation = []
            for i in range(1, len(dialog), 2):  # 偶数索引是患者，奇数索引是医生的回答
                patient = dialog[i-1]["content"]
                doctor = dialog[i]["content"]
                conversation.append({
                    'patient': patient,
                    'doctor': doctor
                })
            conversations.extend(conversation)
    return conversations

train_path = "./data/Chats/Output-1/"
test_path = "./data/SMILE-ChatDdata/"
train_file_paths = [f for f in os.listdir(train_path) if f.endswith(".json")]
test_file_paths = [f for f in os.listdir(test_path) if f.endswith(".json")]

train_data = load_conversations(train_path, train_file_paths)
test_data = load_conversations(test_path, test_file_paths)

train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)





from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

# 加载模型和tokenizer
model_name = "qwen-model"  # 替换为你下载的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# LoRA配置
lora_config = LoraConfig(
    r=8,  # LoRA的rank参数
    lora_alpha=16,  # LoRA的alpha参数
    lora_dropout=0.1,  # LoRA的dropout
    target_modules=["q_proj", "k_proj", "v_proj"],  # LoRA调整的模块
    lora_scale=32  # LoRA缩放系数
)

# 使用LoRA微调模型
model = get_peft_model(model, lora_config)

# 准备训练数据
def preprocess_function(examples):
    # 将对话内容拼接起来以适应模型输入
    input_texts = [f"下面是心理健康医生和病患的对话，你需要学习医生的共情对话能力。{item['patient']};{item['doctor']}" for item in examples['patient']]
    return tokenizer(input_texts, truncation=True, padding="max_length", max_length=512)

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)





from transformers import Trainer, TrainingArguments

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

# 创建Trainer对象
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

# 开始训练
trainer.train()




from datasets import load_metric

# 加载BLEU和Distinct指标
bleu = load_metric("bleu")
distinct = load_metric("distinct")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # 计算BLEU-1/2/3
    bleu_score = bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
    
    # 计算Distinct-1/2
    distinct_1 = distinct.compute(predictions=decoded_preds, n=1)
    distinct_2 = distinct.compute(predictions=decoded_preds, n=2)
    
    return {
        "bleu-1": bleu_score["bleu"],
        "distinct-1": distinct_1["distinct"],
        "distinct-2": distinct_2["distinct"],
    }

# 评估模型
trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="eval", compute_metrics=compute_metrics)
