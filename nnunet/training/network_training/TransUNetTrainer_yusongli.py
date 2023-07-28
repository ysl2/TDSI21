import contextlib
from nnunet.training.network_training.TransUNetTrainer import TransUNetTrainer

class TransUNetTrainer_yusongli(TransUNetTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_num_epochs = 1000

    def load_plans_file(self):
        super().load_plans_file()
        batch_size = 2
        self.plans['plans_per_stage'][0]['batch_size'] = batch_size
        with contextlib.suppress(Exception):
            self.plans['plans_per_stage'][1]['batch_size'] = batch_size
