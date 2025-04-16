from pathlib import Path
from abc import ABC, abstractmethod

class Renderer(ABC):
    def __init__(self, scheduler, render_info_path: Path):
        self.scheduler = scheduler
        self.render_info_path = render_info_path

    @abstractmethod
    def render(self, title: str = "Renderer"):
        raise NotImplementedError("You should implement this method.")