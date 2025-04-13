import tensorflow as tf
from tensorflow.keras import layers, models

def build_custom_model(input_shape=(224, 224, 3), num_classes=80, max_boxes=10):
    model_input = layers.Input(shape=input_shape)

    # Feature extraction layers
    x = layers.Conv2D(32, (3, 3), activation='relu')(model_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    # Output for classification
    class_output = layers.Dense(num_classes, activation='softmax', name='class_output')(x)

    # Output for bounding box regression with max_boxes
    bbox_dense_output = layers.Dense(max_boxes * 4, activation='linear', name='bbox_dense_output')(x)
    bbox_output = layers.Reshape((max_boxes, 4), name='bbox_output')(bbox_dense_output)

    model = models.Model(inputs=model_input, outputs=[class_output, bbox_output])

    model.compile(optimizer='adam', 
                  loss={'class_output': 'categorical_crossentropy', 'bbox_output': 'mean_squared_error'}, 
                  metrics={'class_output': ['accuracy'], 'bbox_output': ['mean_squared_error']})

    return model