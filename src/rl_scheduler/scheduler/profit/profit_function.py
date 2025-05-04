class ProfitFunction:
    def __init__(
        self, job_instance_id: int, price: float, deadline: int, late_penalty: float
    ):
        """
        :param price: 기본 가격
        :param deadline: 마감 기한
        :param late_penalty: 마감 초과 시 발생하는 페널티 (단위 시간당)
        """
        self.job_instance_id = job_instance_id
        self.price = price
        self.deadline = deadline
        self.late_penalty = late_penalty

    def __call__(self, time: int) -> float:
        """
        기대 수익을 계산하는 함수.
        :param time: 작업이 완료된 시간
        :return: 기대 수익
        """
        if time <= self.deadline:
            return self.price
        else:
            late_time = time - self.deadline
            return self.price - self.late_penalty * late_time

    def __str__(self):
        return f"""ProfitFunction(job_instance_id={self.job_instance_id},
        price={self.price}, deadline={self.deadline}, late_penalty=
        {self.late_penalty})"""


# # 사용 예시
# profit_fn = ProfitFunction(price=1000.0, deadline=15.0, late_penalty=50.0)
# print(profit_fn(10))  # 1000.0 (제시간에 완료)
# print(profit_fn(15))  # 1000.0 (마감일에 완료)
# print(profit_fn(20))  # 750.0 (5시간 초과, 패널티 50*5 = 250 차감)
# print(profit_fn(40))  # 0.0 (수익이 0 이하로 떨어지면 0으로 보정)
