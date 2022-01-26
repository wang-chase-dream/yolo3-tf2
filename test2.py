from functools import partial

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam

from nets.yolo import get_train_model, yolo_body
from utils.callbacks import (ExponentDecayScheduler, LossHistory,
                             ModelCheckpoint)
from utils.dataloader import YoloDatasets
from utils.utils import get_anchors, get_classes
from utils.utils_fit import fit_one_epoch



model = get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask)