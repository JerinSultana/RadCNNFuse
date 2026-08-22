import logging
import numpy as np
import SimpleITK as sitk

from radiomics import featureextractor

from .preprocessing import (
    preprocess_for_radiomics,
    get_roi_mask
)


logging.getLogger("radiomics").setLevel(
    logging.ERROR
)


RADIOMICS_SETTINGS = {
    "normalize": True,
    "normalizeScale": 100,
    "resampledPixelSpacing": [1.5, 1.5],
    "interpolator": "sitkBSpline",
    "force2D": True,
    "force2Ddimension": 0,
    "label": 1,
    "binWidth": 20,
    "additionalInfo": True
}


EXPECTED_RADIOMICS_FEATURES = 567


def create_radiomics_extractor():
    """Create and configure the PyRadiomics extractor."""

    extractor = featureextractor.RadiomicsFeatureExtractor(
        **RADIOMICS_SETTINGS
    )

    extractor.disableAllImageTypes()

    extractor.enableImageTypeByName("Original")
    extractor.enableImageTypeByName("Wavelet")
    extractor.enableImageTypeByName("LoG")
    extractor.enableImageTypeByName("Square")
    extractor.enableImageTypeByName("SquareRoot")
    extractor.enableImageTypeByName("Exponential")

    extractor.disableAllFeatures()

    extractor.enableFeatureClassByName("FirstOrder")
    extractor.enableFeatureClassByName("GLCM")
    extractor.enableFeatureClassByName("GLDM")
    extractor.enableFeatureClassByName("GLRLM")
    extractor.enableFeatureClassByName("GLSZM")
    extractor.enableFeatureClassByName("NGTDM")
    extractor.enableFeatureClassByName("Shape2D")

    return extractor


def extract_radiomics_features(
    image_path,
    mask_path=None,
    extractor=None,
    image_size=(224, 224)
):
    """
    Extract radiomics features from one medical image.

    Returns
    -------
    radiomics_features : np.ndarray
        Radiomics feature vector with 567 features.

    mask_source : str
        'external' or 'automatic'.
    """

    if extractor is None:
        extractor = create_radiomics_extractor()

    image = preprocess_for_radiomics(
        image_path,
        image_size
    )

    mask, mask_source = get_roi_mask(
        image_path,
        mask_path,
        image_size
    )

    if mask is None:
        raise ValueError(
            "ROI mask generation failed."
        )

    if np.count_nonzero(mask) == 0:
        raise ValueError(
            "No labels found in this mask "
            "(i.e. nothing is segmented)!"
        )

    sitk_image = sitk.GetImageFromArray(
        image.astype(np.float32)
    )

    sitk_mask = sitk.GetImageFromArray(
        mask.astype(np.uint8)
    )

    result = extractor.execute(
        sitk_image,
        sitk_mask
    )

    feature_dict = {}

    for key, value in result.items():

        if str(key).startswith("diagnostics_"):
            continue

        try:

            numeric_value = float(value)

            if np.isfinite(numeric_value):

                feature_dict[str(key)] = numeric_value

        except (
            TypeError,
            ValueError
        ):
            continue

    if len(feature_dict) != EXPECTED_RADIOMICS_FEATURES:

        raise ValueError(
            "Unexpected radiomics feature dimension: "
            f"{len(feature_dict)}. "
            f"Expected {EXPECTED_RADIOMICS_FEATURES}."
        )

    radiomics_features = np.asarray(
        list(feature_dict.values()),
        dtype=np.float64
    )

    return (
        radiomics_features,
        mask_source
    )
