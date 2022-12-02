from torch.utils.tensorboard import SummaryWriter
import os


class Logger:
    def __init__(self,
                 config: dict):

        log_file_name = '__'.join([f'{k}:{v}' for k, v in config.items()])
        log_file_path = os.path.join(LOG_DIR, log_file_name)
        self._summ_writer = SummaryWriter(log_file_path)

    def log_scalar(self, scalar, name, step_):
        self._summ_writer.add_scalar('{}'.format(name), scalar, step_)

    def log_video(self, video_frames, name, step, fps=10):
        assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
        self._summ_writer.add_video('{}'.format(name), video_frames, step, fps=fps)
