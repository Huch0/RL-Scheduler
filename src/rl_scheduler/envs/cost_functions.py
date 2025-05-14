from rl_scheduler.scheduler import Scheduler


def compute_costs(scheduler: Scheduler) -> dict[str, float]:
    machine_uptime = [
        sum(op.duration for op in machine.assigned_operations)
        for machine in scheduler.machine_instances
    ]
    machine_end_time = [
        machine.last_assigned_end_time for machine in scheduler.machine_instances
    ]
    machine_idle_time = [
        machine_end_time[i] - machine_uptime[i] for i in range(len(machine_uptime))
    ]
    return {
        "C_max": max(machine_end_time),
        "C_mup": sum(machine_uptime),
        "C_mid": sum(machine_idle_time),
    }
