from abc import ABC, abstractmethod
from ..machine import MachineInstance
from ..operation import OperationInstance


class SlotAllocator(ABC):
    @staticmethod
    @abstractmethod
    def find_and_allocate_slot(
        machine_instance: MachineInstance, operation_instance: OperationInstance
    ):
        """
        Abstract method to find and allocate a slot for the given operation
        in the machine.
        """
        pass
