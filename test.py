import json
from datasets import Dataset

# 假设数据集 A 和 B 存储为 JSON 文件
def load_conversation_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 加载数据集 A 和数据集 B
dataset_A = load_conversation_data('path_to_data_A.json')
dataset_B = load_conversation_data('path_to_data_B.json')

# 将数据转换为 datasets 格式
def convert_to_dataset_format(data):
    conversations = []
    conversation = []
    for entry in data:
        role = entry['role']
        content = entry['content']
        if role == "医生":  # 假设我们需要医生的回答
            conversation.append(content)
            conversations.append(conversation)
            conversation = []
        else:
            conversation.append(content)
    return Dataset.from_dict({'conversation': conversations})

# 转换为 dataset 格式
train_dataset = convert_to_dataset_format(dataset_A)
test_dataset = convert_to_dataset_format(dataset_B)




from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# 本地模型路径
model_path = 'model'  # 模型存储在本地路径 model 下
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 对话数据的预处理函数
def preprocess_function(examples):
    inputs = []
    for conversation in examples['conversation']:
        input_text = " ".join(conversation)  # 组合整个对话内容
        inputs.append(input_text)
    model_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    return model_inputs

# 对数据集进行预处理
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# 训练参数设置
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",  # 每个 epoch 后进行评估
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir='./logs',
)

# 使用 Trainer 进行微调
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# 开始训练
trainer.train()




from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter

# 计算 BLEU 分数的函数
def compute_bleu(predictions, references):
    bleu_1, bleu_2, bleu_3 = 0, 0, 0
    for pred, ref in zip(predictions, references):
        bleu_1 += sentence_bleu([ref], pred, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
        bleu_2 += sentence_bleu([ref], pred, weights=(0, 1, 0, 0), smoothing_function=SmoothingFunction().method1)
        bleu_3 += sentence_bleu([ref], pred, weights=(0, 0, 1, 0), smoothing_function=SmoothingFunction().method1)
    return bleu_1 / len(predictions), bleu_2 / len(predictions), bleu_3 / len(predictions)

# 计算 Distinct 分数的函数
def compute_distinct(predictions):
    def n_grams(text, n):
        words = text.split()
        return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    
    def distinct_score(n, predictions):
        n_grams_all = []
        for pred in predictions:
            n_grams_all.extend(n_grams(pred, n))
        distinct_n = len(set(n_grams_all)) / len(n_grams_all) if len(n_grams_all) > 0 else 0
        return distinct_n
    
    distinct_1 = distinct_score(1, predictions)
    distinct_2 = distinct_score(2, predictions)
    distinct_3 = distinct_score(3, predictions)
    
    return distinct_1, distinct_2, distinct_3

# 获取微调后的模型的预测结果
def generate_responses(dataset, model, tokenizer):
    predictions = []
    for conversation in dataset['conversation']:
        question = conversation[-2]  # 假设最后一轮是医生的回答
        inputs = tokenizer(f"Question: {question}", return_tensors="pt").to(model.device)
        outputs = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(response)
    return predictions

# 获取测试集 B 中的参考答案
references = [conversation[-1] for conversation in test_dataset['conversation']]  # 获取医生的答案

# 微调前的预测结果
predictions_before_finetune = generate_responses(test_dataset, model, tokenizer)

# 微调后的预测结果
# 使用微调后的模型进行生成
predictions_after_finetune = generate_responses(test_dataset, model, tokenizer)

# 计算 BLEU 分数
bleu_before = compute_bleu(predictions_before_finetune, references)
bleu_after = compute_bleu(predictions_after_finetune, references)

# 计算 Distinct 分数
distinct_before = compute_distinct(predictions_before_finetune)
distinct_after = compute_distinct(predictions_after_finetune)

print("微调前的 BLEU 分数：", bleu_before)
print("微调后的 BLEU 分数：", bleu_after)

print("微调前的 Distinct 分数：", distinct_before)
print("微调后的 Distinct 分数：", distinct_after)
