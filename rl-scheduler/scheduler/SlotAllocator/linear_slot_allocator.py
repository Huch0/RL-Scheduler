from .slot_allocator import SlotAllocator
from ..Machine import MachineInstance
from ..Operation import OperationInstance


class LinearSlotAllocator(SlotAllocator):
    @staticmethod
    def find_and_allocate_slot(
        machine_instance: MachineInstance, operation_instance: OperationInstance
    ):
        """
        Finds a slot using linear search and allocates the operation to the
        slot if available.
        """
        slot, insert_index = LinearSlotAllocator.find_slot(
            machine_instance, operation_instance
        )
        LinearSlotAllocator.allocate_operation(
            machine_instance, operation_instance, slot, insert_index
        )

    @staticmethod
    def find_slot(
        machine_instance: MachineInstance, operation_instance: OperationInstance
    ):
        """
        Finds an available slot for the operation in the machine instance and
        returns the slot along with the insert index.
        """
        earliest_start_time = operation_instance.earliest_start_time
        duration = operation_instance.duration

        # Sort assigned operations by start_time
        assigned_operations = machine_instance.assigned_operations

        # Check the gap before the first operation
        if assigned_operations:
            first_op = assigned_operations[0]
            if first_op.start_time - earliest_start_time >= duration:
                return {
                    "start_time": earliest_start_time,
                    "end_time": earliest_start_time + duration,
                }, 0

        # Check gaps between consecutive operations
        for i in range(len(assigned_operations) - 1):
            prev_op = assigned_operations[i]
            next_op = assigned_operations[i + 1]
            slot_start = prev_op.end_time
            slot_end = next_op.start_time

            if (
                slot_start >= earliest_start_time
                and (slot_end - slot_start) >= duration
            ):
                return {
                    "start_time": slot_start,
                    "end_time": slot_start + duration,
                }, i + 1

        # Check the slot after the last operation
        if assigned_operations:
            last_op = assigned_operations[-1]
            slot_start = last_op.end_time
            return {
                "start_time": max(slot_start, earliest_start_time),
                "end_time": max(slot_start, earliest_start_time) + duration,
            }, len(assigned_operations)

        # If no operations are assigned, use the earliest start
        return {
            "start_time": earliest_start_time,
            "end_time": earliest_start_time + duration,
        }, 0

    @staticmethod
    def allocate_operation(
        machine_instance: MachineInstance,
        operation_instance: OperationInstance,
        slot,
        insert_index,
    ):
        """
        Allocates the operation to the given slot and inserts it at the
        specified position.
        """
        operation_instance.schedule(
            machine_instance, slot["start_time"], slot["end_time"]
        )

        # Insert the operation at the specified position
        machine_instance.assigned_operations.insert(insert_index, operation_instance)
