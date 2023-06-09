""" Initialize a Tf model wrapper that feed into Model Runner   """

import tensorflow as tf

from easy_tpp.utils import RunnerPhase
from easy_tpp.utils.tf_utils import set_device, set_optimizer

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_v2_behavior()


class TfModelWrapper:
    def __init__(self, model, base_config, model_config, trainer_config):
        """A wrapper class for Tensorflow backends.

        Args:
            model (BaseModel): a TPP model.
            base_config (EasyTPP.Config): basic configs.
            model_config (EasyTPP.Config): model spec configs.
            trainer_config (EasyTPP.Config): trainer spec configs.
        """
        self.model = model
        self.base_config = base_config
        self.model_config = model_config
        self.trainer_config = trainer_config
        set_device(self.trainer_config.gpu)

        # init session and build model
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.model.build_graph()
        if self.model_config.is_training:
            # set up optimizer
            optimizer = self.trainer_config.optimizer
            self.learning_rate = self.trainer_config.learning_rate
            self.opt = set_optimizer(optimizer, self.learning_rate)
            self.train_op = self.opt.minimize(self.model.loss)

        # set up tensorboard
        self.use_tfb = self.trainer_config.use_tfb
        self.train_summary_writer, self.valid_summary_writer = None, None
        if self.use_tfb:
            self.train_summary_writer = tf.summary.FileWriter(self.base_config.spec['tfb_train_dir'])
            self.valid_summary_writer = tf.summary.FileWriter(self.base_config.spec['tfb_valid_dir'])

        # init variable and saver
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def restore(self, ckpt_dir):
        """Load the checkpoint to restore the model.

        Args:
            ckpt_dir (str): path for the checkpoint.
        """
        self.saver.restore(self.sess, ckpt_dir)

    def save(self, ckpt_dir):
        """Save the checkpoint for the model.

        Args:
            ckpt_dir (str): path for the checkpoint.
        """
        self.saver.save(self.sess, ckpt_dir)

    def write_summary(self, epoch, kv_pairs, phase):
        """Write the kv_paris into the tensorboard.

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

            metric_summary = tf.Summary()
            if summary_writer is not None:
                for k, v in kv_pairs.items():
                    if k != 'num_events':
                        metric_summary.value.add(tag=k, simple_value=v)
                        summary_writer.add_summary(metric_summary, epoch)

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
        model = self.model
        sess = self.sess

        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask, type_mask = batch.data.values()

        # set mode to train
        is_training = (phase == RunnerPhase.TRAIN)

        fd = {
            model.time_seqs: time_seqs,
            model.time_delta_seqs: time_delta_seqs,
            model.type_seqs: type_seqs,
            model.batch_non_pad_mask: batch_non_pad_mask,
            model.attention_mask: attention_mask,
            model.type_mask: type_mask,
            model.is_training: is_training

        }

        # Assume we dont do prediction on train set
        pred_dtime, pred_type = None, None
        label_dtime, label_type = time_delta_seqs[:, 1:], type_seqs[:, 1:]

        mask = batch_non_pad_mask[:, 1:]

        if phase in (RunnerPhase.TRAIN, RunnerPhase.VALIDATE):
            # set mode to train
            if is_training:
                _, loss, num_event = sess.run([self.train_op,
                                               model.loss,
                                               model.num_event],
                                              feed_dict=fd)
            else:
                loss, num_event = sess.run([model.loss,
                                            model.num_event],
                                           feed_dict=fd)

                if self.model.event_sampler:
                    pred_dtime, pred_type = sess.run([model.dtime_predict_one_step,
                                                      model.type_predict_one_step],
                                                     feed_dict=fd)
            return loss, num_event, (pred_dtime, pred_type), (label_dtime, label_type), (mask,)
        else:
            pred_dtime, pred_type = sess.run([model.dtime_generation,
                                              model.type_generation],
                                             feed_dict=fd)
            num_steps = pred_dtime.shape[-1]
            label_dtime = time_delta_seqs[:, -num_steps:]
            label_type = type_seqs[:, -num_steps:]
            return (pred_dtime, pred_type), (label_dtime, label_type)
