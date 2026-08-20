import numpy as np

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

from .preprocessing import preprocess_for_cnn


def create_cnn_feature_model(
    image_size=(224, 224)
):
    """
    Create MobileNetV2 feature extractor.

    Returns
    -------
    Model
        MobileNetV2 with Global Average Pooling.
    """

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(
            image_size[0],
            image_size[1],
            3
        )
    )

    base_model.trainable = False

    feature_model = Model(
        inputs=base_model.input,
        outputs=GlobalAveragePooling2D()(
            base_model.output
        )
    )

    return feature_model


def extract_cnn_features(
    image_path,
    model,
    image_size=(224, 224)
):
    """
    Extract MobileNetV2 CNN features from one image.

    Returns
    -------
    np.ndarray
        1280-dimensional CNN feature vector.
    """

    image = preprocess_for_cnn(
        image_path,
        image_size
    )

    image_batch = np.expand_dims(
        image,
        axis=0
    )

    features = model.predict(
        image_batch,
        verbose=0
    )

    return np.asarray(
        features[0],
        dtype=np.float32
    )
