
import cv2
import numpy as np


DEFAULT_IMAGE_SIZE = (224, 224)


def preprocess_for_radiomics(
    image_path,
    image_size=DEFAULT_IMAGE_SIZE
):
    """Prepare a medical image for PyRadiomics."""

    img = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    img = cv2.resize(
        img,
        image_size,
        interpolation=cv2.INTER_AREA
    )

    img = img.astype(np.float32)

    if img.max() > img.min():

        img = (
            (img - img.min())
            /
            (img.max() - img.min())
        )

        img = (
            img * 255
        ).astype(np.uint8)

    else:

        img = np.zeros_like(
            img,
            dtype=np.uint8
        )

    return img


def preprocess_for_cnn(
    image_path,
    image_size=DEFAULT_IMAGE_SIZE
):
    """Prepare a medical image for MobileNetV2."""

    img = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if img is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    img = cv2.resize(
        img,
        image_size,
        interpolation=cv2.INTER_AREA
    )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = img.astype(
        np.float32
    ) / 255.0

    return img


def create_automatic_mask(
    grayscale_image,
    threshold_value=10
):
    """Generate a binary ROI mask."""

    if grayscale_image is None:
        raise ValueError(
            "Input image is None."
        )

    if len(
        grayscale_image.shape
    ) != 2:
        raise ValueError(
            "Input must be a 2D grayscale image."
        )

    _, mask = cv2.threshold(
        grayscale_image,
        threshold_value,
        1,
        cv2.THRESH_BINARY
    )

    return mask.astype(
        np.uint8
    )


def get_roi_mask(
    image_path,
    mask_path=None,
    image_size=DEFAULT_IMAGE_SIZE
):
    """
    Use an external mask if supplied.
    Otherwise generate an automatic ROI mask.
    """

    if mask_path is not None:

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            raise ValueError(
                f"Unable to read mask: {mask_path}"
            )

        mask = cv2.resize(
            mask,
            image_size,
            interpolation=cv2.INTER_NEAREST
        )

        mask = (
            mask > 0
        ).astype(np.uint8)

        source = "external"

    else:

        img = preprocess_for_radiomics(
            image_path,
            image_size
        )

        mask = create_automatic_mask(
            img
        )

        source = "automatic"

    return mask, source
