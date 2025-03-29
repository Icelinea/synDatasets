from utils.parameters import *
from utils.agent import Agent
import random
import json
import pandas as pd


# path
patientBackgroundPath = "data/PatientBackground/processed/scored_anno_pad_dupu_bg1.csv"
patientPersonaPath = "data/Persona/patient.json"
doctorPersonaPath = "data/Persona/doctor.json"
randomOutputPath = "./data/Chats/Random1/Chat"
scoredOutputPath = "./data/Chats/Scored1/Chat"


def one_chatdata_synthesis(selectMethod, patient, doctor, pbdata, epoch, minChats, maxChats):
    chatNum = random.randint(minChats, maxChats)
    
    chats = []
    chats.append(pbdata.to_dict())
    # print(chats)

    history = []

    for i in range(chatNum):
        # 对话
        patientResponse = patient.chat(pbdata['背景'], history, 0.7, seed=42)
        history.append(patientResponse)
        
        doctorResponse = doctor.chat(pbdata['背景'], history, 0.3, seed=128)
        history.append(doctorResponse)

        # print('--- chat ---')
        # print(patientResponse)
        # print(doctorResponse)

        if "EOF" in patientResponse:
            chats.append({"role": "user", "content": "我认为当前我自身的心理健康问题已得到改善"})
            break
        else:
            chats.append({"role": "user", "content": patientResponse})
            
        if "EOF" in doctorResponse:
            chats.append({"role": "assistant", "content": "我认为当前患者的心理健康问题已得到改善"})
            break
        else:
            chats.append({"role": "assistant", "content": doctorResponse})

    # 输出包括患者背景相关信息
    if selectMethod == 'random':
        with open(randomOutputPath + str(epoch) + '.json', 'w', encoding='utf-8') as o:
            json.dump(chats, o, ensure_ascii=False, indent=4)
    elif selectMethod == 'score':
        with open(scoredOutputPath + str(epoch) + '.json', 'w', encoding='utf-8') as o:
            json.dump(chats, o, ensure_ascii=False, indent=4)


def chatdatas(selectMethod, epoches=20, minChats=10, maxChats=20):
    # 加载
    df = None
    patientPersonadata = None
    doctorPersonadata = None
    pbdata = None
    ppdata = None
    dpdata = None

    # 选取患者背景数据
    df = pd.read_csv(patientBackgroundPath)
    if selectMethod == 'score':
        df["加权分数"] = 0.3 * df["志愿者一致性评分"] + 0.6 * df["志愿者质量评分"] + 0.7 * df["专家一致性评分"] + 0.4 * df["专家质量评分"]
        df = df.sort_values(by="加权分数", ascending=False).reset_index(drop=True)
    # print(df.head())
    # print(df.tail())
    
    with open(patientPersonaPath, 'r', encoding='utf-8') as f:
        patientPersonadata = json.load(f)

    with open(doctorPersonaPath, 'r', encoding='utf-8') as f:
        doctorPersonadata = json.load(f)

    # model
    patient = Agent(globalModelPath, globalModelName, "patient", ppdata)
    doctor = Agent(globalModelPath, globalModelName, "doctor", dpdata)

    patient.model_init("", False)
    doctor.model_init("", False)
    
    for epoch in range(epoches):
        # select - random
        if selectMethod == 'random':
            pbdata = df.sample(n=1).iloc[0]
        elif selectMethod == 'score':
            if epoch + 600 > len(df):
                print("可用的基于评分的背景数据已用完")
                break
            pbdata = df.iloc[epoch + 600]
        else: 
            assert False
        ppdata = random.sample(patientPersonadata, 1)[0]
        dpdata = random.sample(doctorPersonadata, 1)[0]
        
        # system prompt
        patientSystemPrompt = "你是一位{}的希望向医生求助心理健康相关问题的患者，意味着{}。在与心理健康医生交流前，你的病患背景是：{}".format(ppdata['类型'], ppdata['解释'], pbdata['背景'])
        doctorSystemPrompt = "你是一位{}的{}专业心理健康医生，在与患者交流时会一直保持着以下的问诊习惯：你的交流风格很{}，{}在恰当的时间和患者进行共情交流的习惯，{}向患者解释一些心理健康领域的专业术语。在交流前，你了解到患者的背景是：{}".format(dpdata["年龄"], dpdata["性别"], dpdata["语言风格"], dpdata["同理心"], dpdata["解释"], pbdata['背景'])

        patient.prompt_init(patientSystemPrompt, False)
        doctor.prompt_init(doctorSystemPrompt, False)
        
        print('----------- Current epoch:', epoch + 768)
        # print(pbdata['背景'])
        one_chatdata_synthesis(selectMethod, patient, doctor, pbdata, epoch + 768, minChats, maxChats)


if __name__ == '__main__':
    # chatdatas('random', 500)
    chatdatas('score', 200)