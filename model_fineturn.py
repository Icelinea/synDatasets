import json
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

from nltk.translate.bleu_score import sentence_bleu
import random
import numpy as np


def load_dialog_data(data_dir, data_name):
    """
    加载多轮对话数据
    """
    dialog_data = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.json'):
            with open(os.path.join(data_dir, file_name), 'r', encoding='utf-8') as f:
                dialog = json.load(f)
                if data_name == "syn":
                    dialog_data.append(dialog[1:])
                elif data_name == "smile":
                    dialog_data.append(dialog)
                else:
                    exit(0)
    return dialog_data


def preprocess_dialog_data(dialog_data):
    """
    处理对话数据为模型可用的输入
    """
    conversations = []
    for dialog in dialog_data:
        patient_dialogs = []
        doctor_dialogs = []
        for message in dialog:
            if message['role'] == 'user' or message['role'] == 'client': # syn-dataset / Smile dataset
                patient_dialogs.append(message['content'])
            elif message['role'] == 'assistant' or message['role'] == 'counselor':
                doctor_dialogs.append(message['content'])
        
        # 将患者的对话和医生的回答配对
        # min(len(patient_dialogs), len(doctor_dialogs))
        for i in range(min(len(patient_dialogs), len(doctor_dialogs))):
            conversations.append((patient_dialogs[i], doctor_dialogs[i]))
    return conversations


class ConversationDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item


def train(model, tokenizer, train_conversations, epochs=3, batch_size=2, learning_rate=5e-5):
    def create_training_data(conversations):
        inputs = []
        labels = []
        for patient, doctor in conversations:
            prompt = f"{patient}\n"
            response = doctor
            inputs.append(prompt)
            labels.append(response)
        return inputs, labels
    
    train_inputs, train_labels = create_training_data(train_conversations)
    
    inputs = tokenizer(train_inputs, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    labels = tokenizer(train_labels, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = labels.input_ids

    train_dataset = ConversationDataset(inputs)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 训练
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        losses = []
        
        for step, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            
            optimizer.zero_grad()

            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item()}")
            losses.append(loss)
            
            # del input_ids, attention_mask, labels, outputs, loss
            torch.cuda.empty_cache()

        # 保存模型
        with open('data/results/epoch' + str(epoch) + '-loss.json', 'w', encoding='utf-8') as o:
            json.dump(losses, o, ensure_ascii=False, indent=4)
        model.save_pretrained(f'./data/Results/epoch_{epoch}')


# BLEU 评分计算
def calculate_bleu(reference, hypothesis, n):
    reference = [ref.split() for ref in reference]
    hypothesis = hypothesis.split()

    weights = tuple([1/n] * n)
    return sentence_bleu(reference, hypothesis, weights=weights)

# Distinct 计算
def calculate_distinct(conversations):
    all_responses = [response for response in conversations]
    all_words = ' '.join(all_responses).split()
    distinct_1 = len(set(all_words)) / len(all_words)
    distinct_2 = len(set(tuple(all_words[i:i+2]) for i in range(len(all_words)-1))) / (len(all_words)-1)
    distinct_3 = len(set(tuple(all_words[i:i+3]) for i in range(len(all_words)-2))) / (len(all_words)-2)
    return distinct_1, distinct_2, distinct_3

def evaluate_model(model, tokenizer, conversations, output_name):
    references = []
    hypotheses = []
    for patient, doctor in conversations:
        prompt = f"现在你是一位专业的心理医生，具有出共情能力和对来访者感受的深刻理解。确保回应流畅且类似人类的对话，字数控制在 10 到 100 个字之间。请为以下的患者提问生成一个回复，不要加上患者的话，只生成你的回复，也不需要添加任何表示身份的前缀。 {patient}"
        response = None
        with torch.no_grad():  # 禁用梯度计算
            modelInputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            generatedIds = model.generate(
                modelInputs.input_ids,
                max_new_tokens=512
            )
            generatedIds = [
                outputIds[len(input_ids):] for input_ids, outputIds in zip(modelInputs.input_ids, generatedIds)
            ]
            response = tokenizer.batch_decode(generatedIds, skip_special_tokens=True)[0]
        # response = generate_response(prompt)
        
        references.append(doctor)
        hypotheses.append(response)

    bleu_1 = np.mean([calculate_bleu([ref], hyp, n=1) for ref, hyp in zip(references, hypotheses)])
    bleu_2 = np.mean([calculate_bleu([ref], hyp, n=2) for ref, hyp in zip(references, hypotheses)])
    bleu_3 = np.mean([calculate_bleu([ref], hyp, n=3) for ref, hyp in zip(references, hypotheses)])
    distinct_1, distinct_2, distinct_3 = calculate_distinct(hypotheses)
    
    print(f"BLEU-1: {bleu_1:.4f}")
    # print(f"Distinct-1: {distinct_1:.4f}")
    # print(f"Distinct-2: {distinct_2:.4f}")
    
    outputs = []
    outputs.append({"bleu-1": bleu_1, "bleu-2": bleu_2, "bleu-3": bleu_3, "distinct-1": distinct_1, "distinct-2": distinct_2, "distinct-3": distinct_3})
    for i in range(len(references)):
        outputs.append({"label": references[i], "output": hypotheses[i]})
    with open('data/Evaluation/before/' + str(output_name) + '.json', 'w', encoding='utf-8') as o:
        json.dump(outputs, o, ensure_ascii=False, indent=4)
    

def main(command, index=0):
    # parameters
    train_path = './data/Chats/Output-1/'
    # test_path = './data/Chats/Output-1/'
    test_path = './data/SMILE-ChatDdata/'
    model_name = "/root/autodl-tmp/Qwen2-7B-Instruct"
    
    # 加载处理数据
    train_dialog = load_dialog_data(train_path, "syn")
    test_dialog = load_dialog_data(test_path, "smile")
    # train_dialog = random.sample(train_dialog, 10)
    test_dialog = random.sample(test_dialog, 10)
    
    train_conversations = preprocess_dialog_data(train_dialog)
    test_conversations = preprocess_dialog_data(test_dialog)

    # 加载预训练模型
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to("cuda")

    if command == "train":
        train(model, tokenizer, train_conversations)
    elif command == "test":
        # 评估模型
        evaluate_model(model, tokenizer, test_conversations, 'smile' + str(index))


if __name__ == '__main__':
    main("train")