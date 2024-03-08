# CustomEnv 소개

## 배경

저희가 정의한 Job shop Scheduling 문제는 고객이 제시한 기준에 부합하는 최적의 스케줄 표를 만드는 것 입니다. 이를 위해 모델은 매 순간 아래 사항을 결정해야합니다.

"어떤 주문의 작업을, 어떤 리소스에 분배할 것인가"

작업의 스케줄링 순서가 조금만 바뀌어도 스케줄 표의 최종 마감시간, 각 리소스의 가동률, 주문 별 마감시간을 지켰는지의 여부 등에 상당히 큰 변화를 겪습니다. 고객이 제시한 조건을 이해하고 최적의 스케줄 표를 찾기 위해, 강화학습 라이브러리 Stable-Baselines3를 활용할 수 있는 커스텀 환경을 만들었습니다.
## 개념 정리

### Resource 

Resource는 Task를 처리할 가상의 공장입니다. 각각의 Resource는 아래와 같은 정보를 갖습니다. 

1. 해당 Resource의 이름
2. 해당 Resource의 처리 가능한 Task type의 집합 (ability)
3. 해당 Resource의 스케줄 표 (task_schedule)
4. 해당 Resource의 가동률 (operation_rate)

각각의 Resoure는 모든 Task를 다 처리할 수는 없습니다. Resource가 어떤 Task의 Type을 처리할 능력이 없다면 해당 Resource의 Task_Schedule에 들어갈 수 없습니다. 

### Order 

Order는 Task들의 집합입니다. 각각의 Task는 아래와 같은 정보를 갖습니다.

1. 해당 Task를 처리하기 위한 duration 정보
2. 해당 Task의 Type
3. Task의 소속 Order에서 해당 Task가 갖는 우선순위

Order는 위 Task들의 정보와 아래의 정보를 갖습니다. 

1. 해당 Order의 이름
2. 시각화를 위한 색상
3. 초기 시작 제약 조건 (Earliest_start)
4. 마감일 (Deadline)
5. 해당 Order의 처리 밀도 (density)

## Definition of State

모델이 관측하는 상태 정보는 아래와 같습니다.

1. Action Mask
2. 각 Order별 Task에 대한 Details
3. 각 Order별 처리 밀도 (실수 배열로 전달)
4. 각 Resource별 가동률 (실수 배열로 전달)
5. 각 Resource별 스케줄링된 Task 개수 (정수 배열로 전달)
6. 각 Resource의 처리 능력 (정수 배열로 전달)
7. 각 Resource의 스케줄표 (0,1로 표현된 2차원 이미지로 전달)

Truncated State : Step이 10000번에 도달한 경우
Terminated State : 모든 Order들의 Task 분배가 끝난 경우
## Definition of Action 

모델의 행동은 아래와 같습니다.

1. Task를 분배할 Resource를 선택
2. 어떤 Order의 Task를 스케줄링 할것인지를 선택

기존에는 2차원으로 Action Space를 구상했으나 Stable-Baselines3의 Deep Q-Network (DQN), Proximal Policy Optimization (PPO), 및 Actor-Critic 등의 알고리즘이 범용적으로 작동할 수 있도록 1차원으로 차원을 축소한 Action Space를 구현했습니다

추가로 MaskablePPO의 동작을 위해 Action Mask도 제공하고 있습니다.
## Definition of Reward

모델이 보상을 얻는 원인은 아래와 같습니다.

1. 매 스텝마다 Illegal action을 한 경우 -0.5점을 받습니다.
2. 매 스텝마다 Legal action을 한 경우 Resource의 가동률에 따라 적정 점수를 받습니다.
3. Terminated State에 도달하면 가장 늦게 끝난 Task의 시간, Order 각각의 Deadline을 준수했는 지 여부, Resource 가동률을 근거로 최종 점수를 받습니다. 

#### 2번 추가 설명

매 스텝에서 Resource들의 가동률을 계산하고 가동률의 평균을 점수로 얻습니다. Resource의 가동률에 대한 계산 방법은 아래와 같습니다.

1. 현재 State에서 가장 늦게 종료된 Task의 종료 시간을 계산합니다.
2. [0, 1번에서 얻은 종료 시간]을 구간으로 하여 해당 리소스의 idle time을 계산합니다.
3. 전체 구간의 길이에서 2번에서 계산한 값을 빼고 전체 구간의 길이로 나눕니다.

2번에 해당하는 보상이 없으면 학습을 하는 동안 어떻게 행동을 선택해야 Terminated_state에 도달하는 지를 알아내지 못 합니다.

#### 3번 추가 설명

최종점수는 100점을 만점으로 합니다. 고객은 평가 기준을 직접 정할 수 있습니다. 고객이 정할 수 있는 기준 및 기본 값은 아래와 같습니다.

1. 가장 늦게 끝난 Task의 종료 시간 - 80%
2. Resource 별 가동률 - 0%
3. Order 별 마감기한 내 종료 여부 - 20%

고객이 목표로 전체 스케줄의 목표 마감 시간을 입력하고 원하는 대로 가중치를 입력하면 최종 점수를 100점을 만점으로 하여 점수를 부여받습니다.


- 랜덤으로 스케줄링한 경우
![[Pasted image 20240304140737.png]]


- 1,000,000번 학습한 PPO 모델의 성능
![[Pasted image 20240304140635.png]]

## 참고 자료
1. MCTS Env
2. JSS Env