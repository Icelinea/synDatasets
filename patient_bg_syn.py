import random
import json
from utils.parameters import *
from utils.agent import Agent

import os
# 设置环境变量来控制 PyTorch 显存管理
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.9'

# Parameters
dialogueTopics = {
    "情绪表现": "指个体在某段时间内的整体情感状态，如愉快、抑郁、焦虑等，影响思维和行为",
    "兴趣爱好": "指个体对活动、事物或人的关注和吸引力。兴趣缺失可能与抑郁等心理问题相关",
    "心理状态": "指个体的思维、情绪、行为和认知功能的总体表现，常用于评估心理健康",
    "睡眠情况": "指休息性状态，影响身心健康。失眠或过度睡眠是常见的心理健康问题",
    "食欲状况": "指个体对食物的渴望或需求，食欲变化可反映抑郁、焦虑等情绪问题",
    "躯体症状": "指由心理因素引起的身体症状，如头痛、胸痛等，常见于心理疾病患者",
    "社交功能": "指个体在社会互动中的适应能力，包括工作、家庭和朋友关系等方面",
    "自杀倾向": "指个体产生自杀念头、计划或行为的倾向，通常与严重的心理疾病相关",
    "其他情况": "个体不存在以上所有普遍心理健康问题的情况"
}

# 患者背景知识生成
# 额外知识 + 输出格式
setupPrompt = "你是一名心理健康领域的专家，以下是你需要理解的在与病患对话时的核心病症的对话主题种类，你所进行的所有的心理健康对话的核心病症的主题都可以被归类到以下的情况当中：{}。你生成的消息一定要按照以下的 json 格式输出，其中 () 内表示你需要生成的内容，生成时请去除 () 并且不要生成多余的空格或者不符合 json 格式或者要求的标点符号，外层是列表符号 []，内层是字典符号 {{}}，具体 json 格式如下：[{{\"年龄\": (), \"性别\": (), \"主题\": (), \"背景\": ()}}]。注意其中年龄为人类的正常年龄，不能小于 0 岁；性别只有'男'和'女'两种；主题是心理健康对话的核心病症的主题的其中一个，需要由你判断所生成的患者背景信息最符合的病症主题，并且只能在提供的情况内选择，有且只有一种主题；背景是与你对话的病患所提供的病情背景信息，需要由你生成，字数不能少于 20 字且不能多于 100 字。".format(dialogueTopics)


def background_synthesis(role, epoches, genNum, exampleNum, dataPath, outputPath):
    assert role == "patient"
    
    data = None
    with open(dataPath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert data != None

    # print(data[0]["案例简述"])  # test

    agent = Agent(globalModelPath, globalModelName, role)
    agent.model_init(setupPrompt)
    
    reponses = []
    for epoch in range(epoches):
        selects = random.sample(data, exampleNum)
        selectData = [dic["案例简述"] for dic in selects]
        reponse = agent.generate(genNum, selectData)

        print(epoch, reponse)
        # mov = json.loads(reponse)
        # for i in mov: reponses.append(i)
        reponses.append(reponse)

    # write 格式的问题，indent？
    with open(outputPath, 'w', encoding='utf-8') as o:
        json.dump(reponses, o, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    background_synthesis("patient", 10, 1, 10, "./data/CPsyCounR/CPsyCounR.json", "./data/PatientBackground/bg1.json")