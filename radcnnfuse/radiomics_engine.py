```python
# ============================================================
# RadCNNFuse
# Radiomics Feature Extraction Engine
# ============================================================

import logging

import numpy as np
import SimpleITK as sitk

from radiomics import featureextractor

from .preprocessing import (
    preprocess_for_radiomics,
    get_roi_mask
)


# ------------------------------------------------------------
# Suppress unnecessary PyRadiomics logging
# ------------------------------------------------------------

logging.getLogger("radiomics").setLevel(
    logging.ERROR
)


# ============================================================
# Radiomics Configuration
# ============================================================

RADIOMICS_SETTINGS = {

    "normalize": True,

    "normalizeScale": 100,

    "resampledPixelSpacing": [
        1.5,
        1.5
    ],

    "interpolator": "sitkBSpline",

    "force2D": True,

    "force2Ddimension": 0,

    "label": 1,

    "binWidth": 20,

    "additionalInfo": True
}


# ------------------------------------------------------------
# Expected radiomics feature dimension
# ------------------------------------------------------------

EXPECTED_RADIOMICS_FEATURES = 567


# ============================================================
# Create Radiomics Extractor
# ============================================================

def create_radiomics_extractor():
    """
    Create and configure the PyRadiomics extractor.

    Returns
    -------
    RadiomicsFeatureExtractor
        Configured PyRadiomics extractor.
    """

    extractor = featureextractor.RadiomicsFeatureExtractor(
        **RADIOMICS_SETTINGS
    )


    # ========================================================
    # Image Types
    # ========================================================

    extractor.disableAllImageTypes()

    extractor.enableImageTypeByName(
        "Original"
    )

    extractor.enableImageTypeByName(
        "Wavelet"
    )

    extractor.enableImageTypeByName(
        "LoG"
    )

    extractor.enableImageTypeByName(
        "Square"
    )

    extractor.enableImageTypeByName(
        "SquareRoot"
    )

    extractor.enableImageTypeByName(
        "Exponential"
    )


    # ========================================================
    # Feature Classes
    #
    # IMPORTANT:
    # PyRadiomics uses these exact class names:
    #
    # firstorder
    # glcm
    # gldm
    # glrlm
    # glszm
    # ngtdm
    # shape2D
    # ========================================================

    extractor.disableAllFeatures()

    extractor.enableFeatureClassByName(
        "firstorder"
    )

    extractor.enableFeatureClassByName(
        "glcm"
    )

    extractor.enableFeatureClassByName(
        "gldm"
    )

    extractor.enableFeatureClassByName(
        "glrlm"
    )

    extractor.enableFeatureClassByName(
        "glszm"
    )

    extractor.enableFeatureClassByName(
        "ngtdm"
    )

    extractor.enableFeatureClassByName(
        "shape2D"
    )


    return extractor


# ============================================================
# Extract Radiomics Features
# ============================================================

def extract_radiomics_features(
    image_path,
    mask_path=None,
    extractor=None,
    image_size=(224, 224)
):
    """
    Extract radiomics features from one medical image.

    Parameters
    ----------
    image_path : str
        Path to medical image.

    mask_path : str, optional
        Optional external ROI mask.

    extractor : RadiomicsFeatureExtractor, optional
        Configured PyRadiomics extractor.

    image_size : tuple
        Target image size.

    Returns
    -------
    radiomics_features : np.ndarray
        Radiomics feature vector with 567 features.

    mask_source : str
        'external' or 'automatic'.
    """


    # --------------------------------------------------------
    # Create extractor if not supplied
    # --------------------------------------------------------

    if extractor is None:

        extractor = create_radiomics_extractor()


    # ========================================================
    # Image preprocessing
    # ========================================================

    image = preprocess_for_radiomics(
        image_path,
        image_size
    )


    # ========================================================
    # ROI / Mask generation
    # ========================================================

    mask, mask_source = get_roi_mask(
        image_path,
        mask_path,
        image_size
    )


    # --------------------------------------------------------
    # Validate mask
    # --------------------------------------------------------

    if mask is None:

        raise ValueError(
            "ROI mask generation failed."
        )


    if np.count_nonzero(mask) == 0:

        raise ValueError(
            "No labels found in this mask "
            "(i.e. nothing is segmented)!"
        )


    # --------------------------------------------------------
    # Validate image and mask dimensions
    # --------------------------------------------------------

    if image.shape != mask.shape:

        raise ValueError(
            "Image and mask dimensions do not match. "
            f"Image: {image.shape}, "
            f"Mask: {mask.shape}"
        )


    # ========================================================
    # Convert NumPy → SimpleITK
    # ========================================================

    sitk_image = sitk.GetImageFromArray(
        image.astype(np.float32)
    )

    sitk_mask = sitk.GetImageFromArray(
        mask.astype(np.uint8)
    )


    # ========================================================
    # PyRadiomics extraction
    # ========================================================

    result = extractor.execute(
        sitk_image,
        sitk_mask
    )


    # ========================================================
    # Collect numeric radiomics features
    # ========================================================

    feature_dict = {}


    for key, value in result.items():

        # ----------------------------------------------------
        # Ignore diagnostic features
        # ----------------------------------------------------

        if str(key).startswith(
            "diagnostics_"
        ):

            continue


        # ----------------------------------------------------
        # Convert feature value to float
        # ----------------------------------------------------

        try:

            numeric_value = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            continue


        # ----------------------------------------------------
        # Remove NaN / Inf
        # ----------------------------------------------------

        if not np.isfinite(
            numeric_value
        ):

            continue


        feature_dict[
            str(key)
        ] = numeric_value


    # ========================================================
    # Validate feature dimension
    # ========================================================

    if len(feature_dict) != EXPECTED_RADIOMICS_FEATURES:

        raise ValueError(
            "Unexpected radiomics feature dimension: "
            f"{len(feature_dict)}. "
            f"Expected "
            f"{EXPECTED_RADIOMICS_FEATURES}."
        )


    # ========================================================
    # Convert to NumPy array
    # ========================================================

    radiomics_features = np.asarray(
        list(
            feature_dict.values()
        ),
        dtype=np.float64
    )


    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if radiomics_features.shape != (
        EXPECTED_RADIOMICS_FEATURES,
    ):

        raise ValueError(
            "Radiomics feature vector has "
            "unexpected shape: "
            f"{radiomics_features.shape}"
        )


    if not np.all(
        np.isfinite(
            radiomics_features
        )
    ):

        raise ValueError(
            "Radiomics feature vector contains "
            "NaN or Inf values."
        )


    # ========================================================
    # Return
    # ========================================================

    return (
        radiomics_features,
        mask_source
    )
```
