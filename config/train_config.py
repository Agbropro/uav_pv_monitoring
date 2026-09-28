import torch

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

NUM_CLASSES = 12

IMAGE_WIDTH = 48
IMAGE_HEIGHT = 80

BATCH_SIZE = 32

LEARNING_RATE = 1e-6

EPOCHS = 300

MODEL_NAME = "mobilenet_v3_small"
