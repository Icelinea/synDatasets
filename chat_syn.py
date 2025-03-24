from utils.parameters import *
from utils.agent import Agent
import random
import json


# prompt
patientPrompt = 1
doctorPrompt = 2


def chatdata_synthesis(epoches, maxChats, outputPath):
    patient = Agent(globalModelPath, globalModelName, "patient")
    doctor = Agent(globalModelPath, globalModelName, "doctor")

    # 具体 prompt?
    patient.model_init(patientPrompt)
    doctor.model_init(doctorPrompt)

    reponses = []
    for epoch in range(epoches):
        chatNum = random.randint(1, maxChats)
        chats = []

        patientResponse = None
        doctorResponse = None
        
        for i in range(chatNum):
            # 对话
            patientResponse = patient.chat(doctorResponse, chats)
            doctorResponse = doctor.chat(patientResponse, chats)
            chats.append(
                {"patient": patientResponse},
                {"doctor": doctorResponse}
            )

    # 输出包括患者背景相关信息(第一个) + 对话信息 ？
    with open(outputPath, 'w', encoding='utf-8') as o:
        json.dump(reponses, o, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    chatdata_synthesis(3, 3, "./data/Chats")