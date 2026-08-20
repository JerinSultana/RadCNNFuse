
```python
import numpy as np

from .radiomics_engine import extract_radiomics_features
from .cnn_engine import (
    create_cnn_feature_model,
    extract_cnn_features
)


def extract_fused_features(
    image_path,
    cnn_model=None,
    mask_path=None,
    image_size=(224, 224)
):
    """
    Extract radiomics and CNN features and
    concatenate them into one fused feature vector.

    Returns
    -------
    dict
        radiomics
        cnn
        fused
        mask_source
    """

    # ---------------------------------------------------------
    # Create CNN model if not provided
    # ---------------------------------------------------------

    if cnn_model is None:

        cnn_model = create_cnn_feature_model(
            image_size=image_size
        )

    # ---------------------------------------------------------
    # Radiomics features
    # ---------------------------------------------------------

    radiomics_features, mask_source = (
        extract_radiomics_features(
            image_path=image_path,
            mask_path=mask_path,
            image_size=image_size
        )
    )

    # ---------------------------------------------------------
    # CNN features
    # ---------------------------------------------------------

    cnn_features = extract_cnn_features(
        image_path=image_path,
        model=cnn_model,
        image_size=image_size
    )

    # ---------------------------------------------------------
    # Fusion
    # ---------------------------------------------------------

    fused_features = np.concatenate(
        [
            radiomics_features,
            cnn_features
        ]
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    if not np.isfinite(
        fused_features
    ).all():

        raise ValueError(
            "NaN or Inf detected in fused features."
        )

    return {
        "radiomics": radiomics_features,
        "cnn": cnn_features,
        "fused": fused_features,
        "mask_source": mask_source
    }


def batch_extract(
    image_paths,
    cnn_model=None,
    image_size=(224, 224),
    verbose=True
):
    """
    Extract fused RadCNNFuse features from
    multiple medical images.

    Parameters
    ----------
    image_paths : list
        List of image paths.

    cnn_model : tensorflow model, optional
        MobileNetV2 feature extraction model.

    image_size : tuple
        Image size used for preprocessing.

    verbose : bool
        Print progress information.

    Returns
    -------
    X : np.ndarray
        Fused feature matrix.

    successful_paths : list
        Successfully processed image paths.

    failed_paths : list
        Failed image paths with error messages.
    """

    # ---------------------------------------------------------
    # Create CNN model only once
    # ---------------------------------------------------------

    if cnn_model is None:

        cnn_model = create_cnn_feature_model(
            image_size=image_size
        )

    feature_rows = []

    successful_paths = []

    failed_paths = []

    total = len(image_paths)

    # ---------------------------------------------------------
    # Process images
    # ---------------------------------------------------------

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        try:

            result = extract_fused_features(
                image_path=image_path,
                cnn_model=cnn_model,
                image_size=image_size
            )

            fused = result["fused"]

            feature_rows.append(
                fused
            )

            successful_paths.append(
                str(image_path)
            )

            if verbose:

                print(
                    f"[{index}/{total}] "
                    f"SUCCESS: {image_path}"
                )

        except Exception as e:

            failed_paths.append(
                {
                    "path": str(image_path),
                    "error": str(e)
                }
            )

            if verbose:

                print(
                    f"[{index}/{total}] "
                    f"FAILED: {image_path}"
                )

                print(
                    f"    Error: {e}"
                )

    # ---------------------------------------------------------
    # Create feature matrix
    # ---------------------------------------------------------

    if len(feature_rows) > 0:

        X = np.vstack(
            feature_rows
        )

    else:

        X = np.empty(
            (0, 0),
            dtype=np.float32
        )

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------

    if X.size > 0:

        if not np.isfinite(X).all():

            raise ValueError(
                "NaN or Inf detected in final feature matrix."
            )

    if verbose:

        print(
            "\n=============================================="
        )

        print(
            "        BATCH EXTRACTION SUMMARY"
        )

        print(
            "=============================================="
        )

        print(
            "Total images:",
            total
        )

        print(
            "Successful:",
            len(successful_paths)
        )

        print(
            "Failed:",
            len(failed_paths)
        )

        print(
            "Feature matrix shape:",
            X.shape
        )

    return (
        X,
        successful_paths,
        failed_paths
    )
```
