import shutil
import tempfile

from cycleformers import CycleTrainer, CycleTrainingArguments
from cycleformers.task_processors import CONLL2003ProcessorConfig, CONLL2003Processor


class TrainerBenchmarkMixin:
    params = [
        "Qwen/Qwen2.5-1.5B"
    ]
    def setup(self):
        task_processor = CONLL2003Processor()
        self.dataset_A, self.dataset_B = task_processor.process()

        self.output_dir = tempfile.mkdtemp()
        self.args = CycleTrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_checkpointing=True,
        )

    def teardown(self):
        shutil.rmtree(self.output_dir)
