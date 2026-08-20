import numpy as np

from .radiomics_engine import (
    create_radiomics_extractor,
    extract_radiomics_features
)

from .cnn_engine import (
    create_cnn_feature_model,
    extract_cnn_features
)


class FeatureExtractor:

    def __init__(
        self,
        image_size=(224, 224)
    ):
        self.image_size = image_size

        self.radiomics_extractor = (
            create_radiomics_extractor()
        )

        self.cnn_model = (
            create_cnn_feature_model(
                image_size=image_size
            )
        )

    def extract(
        self,
        image_path,
        mask_path=None
    ):
        """
        Extract radiomics and CNN features
        and return their fused representation.
        """

        # ----------------------------------------------------
        # Radiomics
        # ----------------------------------------------------

        radiomics_features, mask_source = (
            extract_radiomics_features(
                image_path=image_path,
                mask_path=mask_path,
                extractor=self.radiomics_extractor,
                image_size=self.image_size
            )
        )

        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        cnn_features = extract_cnn_features(
            image_path=image_path,
            model=self.cnn_model,
            image_size=self.image_size
        )

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        fused_features = np.concatenate(
            [
                radiomics_features,
                cnn_features
            ]
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if not np.isfinite(
            fused_features
        ).all():

            raise ValueError(
                f"NaN/Inf detected for image: {image_path}"
            )

        return {
            "radiomics": radiomics_features,
            "cnn": cnn_features,
            "fused": fused_features,
            "mask_source": mask_source
        }
