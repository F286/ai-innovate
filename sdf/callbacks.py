class Callback:
    """
    Base class for creating callbacks to be executed during training.
    """
    def on_epoch_end(self, epoch, model):
        """
        Called at the end of an epoch during training.
        
        Parameters:
        - epoch: The current epoch number.
        - model: The model being trained.
        """
        pass