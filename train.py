import tensorflow as tf
import numpy as np
from model import build_custom_model
from utils import load_coco_data


def buildModel():
    # Load COCO data
    images, class_targets, bbox_targets = load_coco_data()

    # Build the model
    model = build_custom_model()

    # Train the model
    model.fit(images, {'class_output': class_targets, 'bbox_output': bbox_targets}, epochs=5, batch_size=16)

    return model

    # # Save the trained model
    # model.save('models/my_model.keras')
