import json

def set_deadline_from_total_duration(orders):
    for order in orders:
        total_duration = sum(task["duration"] for task in order["tasks"])
        # 각 order의 task들의 duration 합의 2배를 deadline으로 설정
        order["deadline"] = int(total_duration * 1.5)

# 기존 JSON 데이터 읽기
with open("v2-12.json", "r") as file:
    data = json.load(file)

# Deadline 추가
set_deadline_from_total_duration(data["orders"])

# 새로운 JSON 파일로 저장
with open("v2-12-deadline.json", "w") as file:
    json.dump(data, file, indent=4)
