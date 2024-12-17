import os
import logging

def setup_logging(logging_dir):
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    # Define the log file path
    log_file = os.path.join(logging_dir, 'training.log')

    logging.basicConfig(
        level=logging.INFO,  # Log INFO level and above
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # Log to file
            logging.StreamHandler()         # Log to console
        ]
    )

def save_checkpoint(model, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model.save_pretrained(checkpoint_dir)
