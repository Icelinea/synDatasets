import os
import json


# parameters
topics = ["情绪表现", "兴趣爱好", "心理状态", "睡眠情况", "食欲状况", "躯体症状", "社交功能", "自杀倾向", "其他情况"]


def jsons_reorganize(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    print("file numbers: ", len(json_files))

    for json_file in json_files:
        json_file_path = os.path.join(input_dir, json_file)

        with open(json_file_path, "r", encoding="utf-8") as f:
            oridata = json.load(f)
            data = oridata[0]

        if isinstance(data, dict) and "加权分数" in data:
            weighted_score = str(data["加权分数"])
            print(weighted_score)

            # folder_path = os.path.join(output_dir, weighted_score)
            folder_path = output_dir
            # if not os.path.exists(folder_path):
            #     os.makedirs(folder_path)

            existing_files = [f for f in os.listdir(folder_path) if f.startswith(weighted_score)]
            file_index = len(existing_files) + 1

            new_filename = f"[{weighted_score}]_{file_index}.json"
            new_file_path = os.path.join(folder_path, new_filename)

            with open(new_file_path, "w", encoding="utf-8") as f:
                json.dump(oridata, f, ensure_ascii=False, indent=4)


def identity_organize(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    print(len(json_files))
    for json_file in json_files:
        json_file_path = os.path.join(input_dir, json_file)
        
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = None
            try:
                data = json.load(f)
            except Exception as e:
                print("Wrong Json Format :", json_file_path)
                exit(1)
            check = 1
            
            for i in range(len(data)):
                if i == 0: continue
                elif (i % 2 == 1):
                    if not ( (data[i]["role"] == "user") and (data[i]["content"][0:2] == "患者") ):
                        check = 0
                        break
                elif (i % 2 == 0):
                    if not ( (data[i]["role"] == "assistant") and (data[i]["content"][0:2] == "医生") ):
                        check = 0
                        break

            if check:
                weighted_score = str(data[0]["加权分数"])
                folder_path = output_dir

                existing_files = [f for f in os.listdir(folder_path) if f.startswith( '[{}]'.format(weighted_score) )]
                file_index = len(existing_files) + 1

                new_filename = f"[{weighted_score}]_{file_index}.json"
                new_file_path = os.path.join(folder_path, new_filename)

                with open(new_file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)


def topics_check(data_dir):
    json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    print("Dataset Numbers: ", len(json_files))

    topic_nums = [0] * 9
    for json_file in json_files:
        json_file_path = os.path.join(data_dir, json_file)

        with open(json_file_path, "r", encoding="utf-8") as f:
            data = None
            try:
                data = json.load(f)
            except Exception as e:
                print("Wrong Json Format :", json_file_path)
                exit(1)
                
            try:
                idx = topics.index(data[0]["主题"])
                topic_nums[idx] += 1
            except Exception as e:
                print('Wrong Format!')
                print(data)
                exit(0)
    # print
    sum = 0
    for i in range(len(topics)):
        sum += topic_nums[i]
        print(topics[i], ":", topic_nums[i])
    print("Sum :", sum)


if __name__ == '__main__':
    # jsons_reorganize("data/Chats/Scored3/", "data/Chats/Output/")

    identity_organize("data/Chats/Output-2/", "data/Chats/Output/")

    """
    Dataset Numbers:  252
    情绪表现 : 37
    兴趣爱好 : 38
    心理状态 : 36
    睡眠情况 : 50
    食欲状况 : 10
    躯体症状 : 1
    社交功能 : 25
    自杀倾向 : 29
    其他情况 : 26
    """

    topics_check("data/Chats/Output")

    # 打印每种标签数量，判断补充数据 + 找 benchmark