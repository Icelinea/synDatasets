patient_bg_syn.py 为患者背景数据生成代码
bg_process.py 为患者背景数据处理代码，包括去重、填充标签和评分过程

chat_synthesis.py 为对话生成代码
chat_process.py 为对话数据处理代码
chat_evaluation.py 为数据统计代码

model_fineturn.py 为模型微调实验代码

utils 文件夹为基础代码模块
其中 agent.py 保存了大语言模型的基础加载类，
parameters.py 保存了全局变量

photos 为绘图文件夹

data 为数据保存点：
+ Chats 数据为生成的对话数据，其中 Output-1 为论文的对话数据，其余是未处理的老数据
+ CPsyCounR 为在患者背景数据生成时作为提示词样例的参考数据
+ Evaluation 为大语言模型微调前后的效果数据，t1 表示微调前，t2 表示微调后
+ PatientBackground 为患者背景数据，raw 为刚完成生成的数据，others 为测试时使用的数据，processed 的 scored_anno_pad_dupu_bg1.csv、scored_anno_pad_dupu_bg2.csv、scored_anno_pad_dupu_bg3.csv 为完整的患者背景数据
+ Persona 为医生和患者的人格提示词细节数据
+ SMILE-ChatDdata 为从 SMILE 数据集中选取的 100 条作为模型微调效果的展示数据