import os
import json


def jsons_reorganize():
    input_dir = "data/Chats/Scored1/"
    output_dir = "data/Chats/Output/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    print("file numbers: ", len(json_files))

    for json_file in json_files:
        json_file_path = os.path.join(input_dir, json_file)

        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data = data[0]

        if isinstance(data, dict) and "加权分数" in data:
            weighted_score = str(data["加权分数"])
            print(weighted_score)

            # folder_path = os.path.join(output_dir, weighted_score)
            folder_path = output_dir
            # if not os.path.exists(folder_path):
            #     os.makedirs(folder_path)

            existing_files = [f for f in os.listdir(folder_path) if f.startswith(weighted_score)]
            file_index = len(existing_files) + 1

            new_filename = f"{weighted_score}_{file_index}.json"
            new_file_path = os.path.join(folder_path, new_filename)

            with open(new_file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    jsons_reorganize()