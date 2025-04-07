from collections import Counter
import json
import os
from collections import defaultdict

from datasets import Dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from utils.parameters import *


# parameters
train_dataset_path = "./data/Chats/Output-1/"
test_dataset_path = "./data/Chats/Output-1/"


def load_datasets(train_data_path, test_data_path):
    def load_multiple_json_files(directory_path):
        all_data = []
        # 遍历文件夹中的所有文件
        for filename in os.listdir(directory_path):
            if filename.endswith(".json"):
                file_path = os.path.join(directory_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data.extend(data[1:])  # 将数据合并到一个列表中
        return all_data
    
    def preprocess_data(dataset):
        dialogues = []
        current_dialogue = []
        
        for turn in dataset:
            role = turn["role"]
            content = turn["content"]
            current_dialogue.append((role, content))
            
            if role == "医生":
                dialogues.append(current_dialogue)
                current_dialogue = []
        
        return dialogues

    train_data = load_multiple_json_files(train_data_path)
    test_data = load_multiple_json_files(test_data_path)
    return preprocess_data(train_data), preprocess_data(test_data)


class ConversationDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item


def mytrain(train_data_path, test_data_path):
    def format_data(dialogues):
        inputs = []
        labels = []
        for dialogue in dialogues:
            input_text = ""
            output_text = ""
            for role, content in dialogue:
                if role == "患者":
                    input_text += f"患者: {content}\n"
                elif role == "医生":
                    output_text += f"医生: {content}\n"
            inputs.append(input_text)
            labels.append(output_text)
        return inputs, labels
    
    def tokenize_data(inputs, labels):
        input_encodings = tokenizer(inputs, truncation=True, padding=True, return_tensors='pt')
        label_encodings = tokenizer(labels, truncation=True, padding=True, return_tensors='pt')
        input_encodings['labels'] = label_encodings['input_ids']
        return input_encodings

    # Data
    train_data, _ = load_datasets(train_data_path, test_data_path)
    train_inputs, train_labels = format_data(train_data)
    train_encodings = tokenize_data(train_inputs, train_labels)
    train_dataset = ConversationDataset(train_encodings)
    
    tokenizer = AutoTokenizer.from_pretrained(testModelPath)
    model = AutoModelForCausalLM.from_pretrained(testModelPath)
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir='./results',
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # 微调模型
    trainer.train()


def evaluate_bleu(model, tokenizer, test_data):
    def compute_bleu(reference, hypothesis):
        reference = [ref.split() for ref in reference]  # 参考答案
        hypothesis = hypothesis.split()  # 预测答案
        smoothing_function = SmoothingFunction().method1
        return sentence_bleu(reference, hypothesis, smoothing_function=smoothing_function)
        
    model.eval()
    bleu_scores = []
    
    for dialogue in test_data:
        input_text = ""
        for role, content in dialogue:
            if role == "患者":
                input_text += f"患者: {content}\n"
        
        # 生成回答
        inputs = tokenizer(input_text, return_tensors='pt')
        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], max_length=100, num_beams=5, no_repeat_ngram_size=2)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reference_text = dialogue[-1][1]
        
        bleu_score = compute_bleu([reference_text], generated_text)
        bleu_scores.append(bleu_score)
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    return avg_bleu


def evaluate_distinct(model, tokenizer, test_data):
    def compute_distinct_ngrams(texts, n):
        ngram_counter = Counter()
        for text in texts:
            tokens = text.split()
            ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
            ngram_counter.update(ngrams)
        distinct_score = len(ngram_counter) / len(texts)
        return distinct_score
        
    model.eval()
    distinct_1_scores, distinct_2_scores, distinct_3_scores = [], [], []
    for dialogue in test_data:
        input_text = ""
        for role, content in dialogue:
            if role == "患者":
                input_text += f"患者: {content}\n"
        
        # 生成回答
        inputs = tokenizer(input_text, return_tensors='pt')
        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], max_length=100, num_beams=5, no_repeat_ngram_size=2)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        distinct_1 = compute_distinct_ngrams([generated_text], 1)
        distinct_2 = compute_distinct_ngrams([generated_text], 2)
        distinct_3 = compute_distinct_ngrams([generated_text], 3)
        
        distinct_1_scores.append(distinct_1)
        distinct_2_scores.append(distinct_2)
        distinct_3_scores.append(distinct_3)
    
    avg_distinct_1 = sum(distinct_1_scores) / len(distinct_1_scores)
    avg_distinct_2 = sum(distinct_2_scores) / len(distinct_2_scores)
    avg_distinct_3 = sum(distinct_3_scores) / len(distinct_3_scores)
    
    return avg_distinct_1, avg_distinct_2, avg_distinct_3


def mytest(train_data_path, test_data_path):
    _, test_data = load_datasets(train_data_path, test_data_path)
    
    tokenizer = AutoTokenizer.from_pretrained(testModelPath)
    model = AutoModelForCausalLM.from_pretrained(testModelPath)
    print(test_data[:3])
    
    bleu_score = evaluate_bleu(model, tokenizer, test_data)
    distinct_score = evaluate_distinct(model, tokenizer, test_data)

    print(bleu_score)
    print(distinct_score)


def main(command, train_data_path, test_data_path):
    if command == "train":
        mytrain(train_data_path, test_data_path)
    elif command == "test":
        mytest(train_data_path, test_data_path)


if __name__ == '__main__':
    main("test", train_dataset_path, test_dataset_path)