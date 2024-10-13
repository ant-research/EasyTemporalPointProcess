""" Initialize a Pytorch model wrapper that feed into Model Runner   """

import torch
from torch.utils.tensorboard import SummaryWriter

from easy_tpp.utils import RunnerPhase, set_optimizer, set_device


class TorchModelWrapper:
    def __init__(self, model, base_config, model_config, trainer_config):
        """A wrapper class for Torch backends.

        Args:
            model (BaseModel): a TPP model.
            base_config (EasyTPP.Config): basic configs.
            model_config (EasyTPP.ModelConfig): model spec configs.
            trainer_config (EasyTPP.TrainerConfig): trainer spec configs.
        """
        self.model = model
        self.base_config = base_config
        self.model_config = model_config
        self.trainer_config = trainer_config

        self.model_id = self.base_config.model_id
        self.device = set_device(self.trainer_config.gpu)

        self.model.to(self.device)

        if self.model_config.is_training:
            # set up optimizer
            optimizer = self.trainer_config.optimizer
            self.learning_rate = self.trainer_config.learning_rate
            self.opt = set_optimizer(optimizer, self.model.parameters(), self.learning_rate)

        # set up tensorboard
        self.use_tfb = self.trainer_config.use_tfb
        self.train_summary_writer, self.valid_summary_writer = None, None
        if self.use_tfb:
            self.train_summary_writer = SummaryWriter(log_dir=self.base_config.spec['tfb_train_dir'])
            self.valid_summary_writer = SummaryWriter(log_dir=self.base_config.spec['tfb_valid_dir'])

    def restore(self, ckpt_dir):
        """Load the checkpoint to restore the model.

        Args:
            ckpt_dir (str): path for the checkpoint.
        """

        self.model.load_state_dict(torch.load(ckpt_dir), strict=False)

    def save(self, ckpt_dir):
        """Save the checkpoint for the model.

        Args:
            ckpt_dir (str): path for the checkpoint.
        """
        torch.save(self.model.state_dict(), ckpt_dir)

    def write_summary(self, epoch, kv_pairs, phase):
        """Write the kv_paris into the tensorboard

        Args:
            epoch (int): epoch index in the training.
            kv_pairs (dict): metrics dict.
            phase (RunnerPhase): a const that defines the stage of model runner.
        """
        if self.use_tfb:
            summary_writer = None
            if phase == RunnerPhase.TRAIN:
                summary_writer = self.train_summary_writer
            elif phase == RunnerPhase.VALIDATE:
                summary_writer = self.valid_summary_writer
            elif phase == RunnerPhase.PREDICT:
                pass

            if summary_writer is not None:
                for k, v in kv_pairs.items():
                    if k != 'num_events':
                        summary_writer.add_scalar(k, v, epoch)

                summary_writer.flush()
        return

    def close_summary(self):
        """Close the tensorboard summary writer.
        """
        if self.train_summary_writer is not None:
            self.train_summary_writer.close()

        if self.valid_summary_writer is not None:
            self.valid_summary_writer.close()
        return

    def run_batch(self, batch, phase):
        """Run one batch.

        Args:
            batch (EasyTPP.BatchEncoding): preprocessed batch data that go into the model.
            phase (RunnerPhase): a const that defines the stage of model runner.

        Returns:
            tuple: for training and validation we return loss, prediction and labels;
            for prediction we return prediction.
        """

        batch = batch.to(self.device).values()
        if phase in (RunnerPhase.TRAIN, RunnerPhase.VALIDATE):
            # set mode to train
            is_training = (phase == RunnerPhase.TRAIN)
            self.model.train(is_training)

            # FullyRNN needs grad event in validation stage
            grad_flag = is_training if not self.model_id == 'FullyNN' else True
            # run model
            with torch.set_grad_enabled(grad_flag):
                loss, num_event = self.model.loglike_loss(batch)

            # Assume we dont do prediction on train set
            pred_dtime, pred_type, label_dtime, label_type, mask = None, None, None, None, None

            # update grad
            if is_training:
                self.opt.zero_grad()
                (loss / num_event).backward()
                self.opt.step()
            else:  # by default we do not do evaluation on train set which may take a long time
                if self.model.event_sampler:
                    self.model.eval()
                    with torch.no_grad():
                        if batch[1] is not None and batch[2] is not None:
                            label_dtime, label_type = batch[1][:, 1:].cpu().numpy(), batch[2][:, 1:].cpu().numpy()
                        if batch[3] is not None:
                            mask = batch[3][:, 1:].cpu().numpy()
                        pred_dtime, pred_type = self.model.predict_one_step_at_every_event(batch=batch)
                        pred_dtime = pred_dtime.detach().cpu().numpy()
                        pred_type = pred_type.detach().cpu().numpy()
            return loss.item(), num_event, (pred_dtime, pred_type), (label_dtime, label_type), (mask,)
        else:
            pred_dtime, pred_type, label_dtime, label_type = self.model.predict_multi_step_since_last_event(batch=batch)
            pred_dtime = pred_dtime.detach().cpu().numpy()
            pred_type = pred_type.detach().cpu().numpy()
            label_dtime = label_dtime.detach().cpu().numpy()
            label_type = label_type.detach().cpu().numpy()
            return (pred_dtime, pred_type), (label_dtime, label_type)
