import lightning.pytorch as pl
from lightning.pytorch import callbacks as callbacks
from lightning.pytorch.loggers import TensorBoardLogger


class MyCallback(pl.Callback):
    def __init__(self, monitor_metric, patience, mode):
        super().__init__()

        self.early_stop = callbacks.early_stopping.EarlyStopping(
            monitor=monitor_metric, mode=mode, min_delta=0.001, patience=patience
        )

        self.lr_monitoring = callbacks.LearningRateMonitor(logging_interval="epoch")

        self.progress_bar = callbacks.RichProgressBar(
            refresh_rate=1,  # number of batches
            leave=True,
            theme=callbacks.progress.rich_progress.RichProgressBarTheme(
                description="green_yellow",
                progress_bar="green1",
                progress_bar_finished="#e0bf06",
                progress_bar_pulse="#0606e0",
                batch_progress="green_yellow",
                time="grey82",
                processing_speed="grey82",
                metrics="grey82",
            ),
        )

        self.model_summary = callbacks.RichModelSummary(max_depth=2)

        # self.checkpoint = callbacks.ModelCheckpoint(
        #     dirpath = "/home/jupyter/reco_project/save_checkpoints/",
        #     filename='Neumf_stg_{}_checkpoint_{epoch}-{val_loss:.2f}'.format(self.strategy),
        #     monitor='val_loss',
        #     save_top_k=1,
        #     mode='min',
        #     every_n_train_steps=None,
        #     every_n_epochs=1,
        #     train_time_interval=None,
        #     enable_version_counter=True
        # )

    def get_callbacks(self):
        return [
            self.early_stop,
            self.lr_monitoring,
            self.progress_bar,
            self.model_summary,
        ]


def get_logger(path_tensor, version, hp_metric):
    """
    # default logger used by trainer (if tensorboard is installed)
    """
    logger = TensorBoardLogger(
        save_dir=path_tensor,
        name="lightning_logs",
        version=version,
        default_hp_metric=hp_metric,
    )

    return logger
