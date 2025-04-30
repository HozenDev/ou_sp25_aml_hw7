'''
Callback for saving a model in a new file each time performance drops below a specified threshold

Developed with ChatGPT
'''
from keras.callbacks import Callback

class ThresholdStepCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', thresholds=[0.05, 0.03, 0.02], verbose=True):
        """
        Save model each time the monitored metric crosses below a new threshold (descending list).
        
        :param filepath: str, formatable string with {epoch} and {metric}
        :param monitor: str, the metric to monitor
        :param thresholds: list of descending thresholds to trigger checkpoints
        :param verbose: whether to print when a checkpoint is saved
        """
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.thresholds = sorted(thresholds, reverse=True)
        self.verbose = verbose
        self.current_idx = 0  # current index in the threshold list to check for

    def on_epoch_end(self, epoch, logs=None):
        '''
        Callback method

        :param epoch: Current epoch
        :param logs: Dict containing all of the logged data
        '''
        
        logs = logs or {}
        
        # Fetch the current value of the monitor variable
        metric_value = logs.get(self.monitor)

        if metric_value is None:
            return

        # Continue saving as long as we haven't used all thresholds
        if self.current_idx < len(self.thresholds):
            # The current threshold that we are searching for
            target = self.thresholds[self.current_idx]
            
            if metric_value < target:
                # Save model
                fname = self.filepath.format(epoch=epoch, metric=self.thresholds[self.current_idx])
                self.model.save(fname)
                if self.verbose:
                    print(f"\nCheckpoint: {self.monitor} dropped below {target:.4f}. Saved model to {fname}")
                self.current_idx += 1  # move to next (lower) threshold
