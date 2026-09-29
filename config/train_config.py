import torch

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

NUM_CLASSES = 12

IMAGE_WIDTH = 72
IMAGE_HEIGHT = 120

BATCH_SIZE = 64

LEARNING_RATE = 0.00014050856870093735
WEIGHT_DECAY = 0.0013048816325475594
FREEZE_BACKBONE = False

EPOCHS = 300

MODEL_NAME = "mobilenet_v3_small"
