import csv
import os
import sys
from torch.utils.tensorboard import SummaryWriter


class CustomLogger(SummaryWriter):
    def __init__(self, log_dir='./experiments/tmp/1', print_to_stdout=True):
        super().__init__(log_dir)
        self.csv_file = os.path.join(log_dir, 'results.csv')
        self.csv_data = {}
        self.current_step = None
        self.print_to_stdout = print_to_stdout

    def add_scalar(self, tag, scalar_value, global_step=None):
        super().add_scalar(tag, scalar_value, global_step)
        self._add_to_csv(tag, scalar_value, global_step)
        # self._print_to_stdout(tag, scalar_value, global_step)

    def add_text(self, tag, text_string, global_step=None):
        super().add_text(tag, text_string, global_step)
        self._add_to_csv(tag, text_string, global_step)
        # self._print_to_stdout(tag, text_string, global_step)

    def _add_to_csv(self, tag, value, global_step):
        if global_step is not None:
            self.current_step = global_step

        if self.current_step not in self.csv_data:
            self.csv_data[self.current_step] = {}

        self.csv_data[self.current_step][tag] = value

    def _print_to_stdout(self, tag, value, global_step):
        if self.print_to_stdout:
            print(f"Step {global_step}: {tag} = {value}", file=sys.stdout)

    def _write_to_csv(self):
        fieldnames = ['global_step'] + sorted(set(key for step in self.csv_data.values() for key in step.keys()))

        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for step, data in sorted(self.csv_data.items()):
                row = {'global_step': step, **data}
                writer.writerow(row)

    def close(self):
        self._write_to_csv()
        super().close()

    def print_summary(self, global_step=None):
        """Print a summary of all logged values for a given step"""
        if global_step is None:
            global_step = self.current_step

        if global_step in self.csv_data:
            print(f"\nSummary for Step {global_step}:")
            for tag, value in sorted(self.csv_data[global_step].items()):
                print(f"  {tag}: {value}")
        else:
            print(f"No data available for Step {global_step}")
