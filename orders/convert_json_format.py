import json

# 불러올 JSON 파일의 경로 및 이름을 지정합니다.
input_file_path = "./orders-default.json"

# JSON 파일을 불러옵니다.
with open(input_file_path, "r") as input_file:
    original_json = json.load(input_file)

def convert_json_format(original_json):

    for order in original_json["orders"]:
        converted_order = {
            "name": order["name"],
            "color": order["color"],
            "earliest_start": order["earliest_start"],
            "tasks": []
        }

        for step in order["steps"]:
            converted_step = {
                "index": step["step"],
                "type": "A",
                "duration": step["duration"],
                "predecessor": step["predecessor"]
            }
            converted_order["tasks"].append(converted_step)

        converted_json["orders"].append(converted_order)

    return converted_json

# convert_json_format 함수를 이용하여 변환된 Json을 얻습니다.
converted_json = convert_json_format(original_json)

output_file_path = "./converted_orders_default.json"

# JSON 파일로 저장합니다.
with open(output_file_path, "w") as output_file:
    json.dump(converted_json, output_file, indent=2)
