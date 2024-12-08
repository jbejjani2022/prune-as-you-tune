import csv
from transformers import TrainerCallback
import os


class LoggerCallback(TrainerCallback):
    """Saves training log data to CSV and saves model checkpoint each epoch."""
    
    def __init__(self, log_file, checkpoint_dir):
        self.log_file = log_file
        self.checkpoint_dir = checkpoint_dir
        # Write CSV header
        with open(self.log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "eval_accuracy", "eval_loss"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # Check if the current log contains the required metrics after each epoch
        if "epoch" in logs and "eval_accuracy" in logs and "eval_loss" in logs:
            epoch = logs.get("epoch")
            eval_accuracy = logs.get("eval_accuracy")
            eval_loss = logs.get("eval_loss")

            # Write the metrics to the CSV file
            with open(self.log_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([epoch, eval_accuracy, eval_loss])

    def on_epoch_end(self, args, state, control, **kwargs):
        # Save a full checkpoint (model, optimizer, scheduler, etc.) at the end of each epoch
        epoch = int(state.epoch)
        checkpoint_path = os.path.join(self.checkpoint_dir, f"ckpt-epoch-{epoch}")
        
        # Save the entire trainer state, which includes model, optimizer, and scheduler states
        kwargs["model"].save_pretrained(checkpoint_path)  # Save model
        state.save_to_json(os.path.join(checkpoint_path, "trainer_state.json"))  # Save trainer state (includes optimizer, scheduler)
        print(f"Epoch {epoch} checkpoint saved at {checkpoint_path}, NOTE: loading from checkpoings perilous with curlora")
