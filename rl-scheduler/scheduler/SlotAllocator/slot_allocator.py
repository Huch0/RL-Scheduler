from abc import ABC, abstractmethod
from ..Machine import MachineInstance
from ..Operation import OperationInstance


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
