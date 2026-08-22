import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .radiomics_engine import (
    create_radiomics_extractor,
    extract_radiomics_features
)

from .cnn_engine import (
    create_cnn_feature_model,
    extract_cnn_features
)


class RadCNNFuse:

    EXPECTED_RADIOMICS_FEATURES = 567
    EXPECTED_CNN_FEATURES = 1280
    EXPECTED_FUSED_FEATURES = 1847

    def __init__(
        self,
        image_size=(224, 224),
        pca_components=100
    ):

        self.image_size = image_size
        self.pca_components = pca_components

        print("Initializing RadCNNFuse...")

        self.radiomics_extractor = (
            create_radiomics_extractor()
        )

        self.cnn_feature_model = (
            create_cnn_feature_model(
                image_size=image_size
            )
        )

        self.scaler = StandardScaler()

        self.pca = PCA(
            n_components=pca_components,
            random_state=42
        )

        print("Radiomics engine: READY")
        print("CNN engine: READY")
        print("Scaler: READY")
        print("PCA: READY")

    # ======================================================
    # Single image
    # ======================================================

    def extract_features(
        self,
        image_path,
        mask_path=None
    ):

        radiomics_features, mask_source = (
            extract_radiomics_features(
                image_path=image_path,
                mask_path=mask_path,
                extractor=self.radiomics_extractor,
                image_size=self.image_size
            )
        )

        if len(radiomics_features) != (
            self.EXPECTED_RADIOMICS_FEATURES
        ):

            raise ValueError(
                "Unexpected radiomics feature dimension: "
                f"{len(radiomics_features)}. "
                f"Expected "
                f"{self.EXPECTED_RADIOMICS_FEATURES}."
            )

        cnn_features = extract_cnn_features(
            image_path=image_path,
            model=self.cnn_feature_model,
            image_size=self.image_size
        )

        if len(cnn_features) != (
            self.EXPECTED_CNN_FEATURES
        ):

            raise ValueError(
                "Unexpected CNN feature dimension: "
                f"{len(cnn_features)}. "
                f"Expected "
                f"{self.EXPECTED_CNN_FEATURES}."
            )

        fused_features = np.concatenate(
            [
                radiomics_features,
                cnn_features
            ]
        )

        if len(fused_features) != (
            self.EXPECTED_FUSED_FEATURES
        ):

            raise ValueError(
                "Unexpected fused feature dimension: "
                f"{len(fused_features)}. "
                f"Expected "
                f"{self.EXPECTED_FUSED_FEATURES}."
            )

        if not np.isfinite(
            fused_features
        ).all():

            raise ValueError(
                "NaN or Inf detected in extracted features."
            )

        return {
            "radiomics": radiomics_features,
            "cnn": cnn_features,
            "fused": fused_features,
            "mask_source": mask_source
        }

    # ======================================================
    # Dataset discovery
    # ======================================================

    def _find_images(
        self,
        image_folder
    ):

        image_folder = Path(
            image_folder
        )

        if not image_folder.exists():

            raise FileNotFoundError(
                f"Image folder not found: {image_folder}"
            )

        extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff"
        }

        image_paths = []

        for path in image_folder.rglob("*"):

            if (
                path.is_file()
                and
                path.suffix.lower()
                in extensions
            ):

                image_paths.append(
                    str(path)
                )

        image_paths.sort()

        if len(image_paths) == 0:

            raise ValueError(
                f"No supported images found in: "
                f"{image_folder}"
            )

        return image_paths

    # ======================================================
    # Full dataset transformation
    # ======================================================

    def transform_dataset(
        self,
        image_folder,
        labels=None,
        output_csv=None
    ):

        image_paths = self._find_images(
            image_folder
        )

        print(
            f"\nFound {len(image_paths)} images."
        )

        print(
            "\nStarting RadCNNFuse extraction..."
        )

        feature_rows = []

        successful_paths = []

        failed_paths = []

        for index, image_path in enumerate(
            image_paths,
            start=1
        ):

            try:

                result = self.extract_features(
                    image_path
                )

                fused = result["fused"]

                feature_rows.append(
                    fused
                )

                successful_paths.append(
                    image_path
                )

                print(
                    f"[{index}/{len(image_paths)}] "
                    f"SUCCESS: "
                    f"{Path(image_path).name}"
                )

            except Exception as e:

                failed_paths.append(
                    {
                        "path": image_path,
                        "error": str(e)
                    }
                )

                print(
                    f"[{index}/{len(image_paths)}] "
                    f"FAILED: "
                    f"{Path(image_path).name}"
                )

                print(
                    f"    Error: {e}"
                )

        if len(feature_rows) == 0:

            raise RuntimeError(
                "No images were successfully processed."
            )

        X = np.asarray(
            feature_rows,
            dtype=np.float32
        )

        print(
            "\nRaw fused feature matrix:",
            X.shape
        )

        if X.shape[1] != (
            self.EXPECTED_FUSED_FEATURES
        ):

            raise ValueError(
                "Unexpected fused feature dimension: "
                f"{X.shape[1]}. "
                f"Expected "
                f"{self.EXPECTED_FUSED_FEATURES}."
            )

        print(
            "\nApplying StandardScaler..."
        )

        X_scaled = self.scaler.fit_transform(
            X
        )

        print(
            "Applying PCA..."
        )

        n_components = min(
            self.pca_components,
            X_scaled.shape[0],
            X_scaled.shape[1]
        )

        if n_components != self.pca_components:

            self.pca = PCA(
                n_components=n_components,
                random_state=42
            )

        X_pca = self.pca.fit_transform(
            X_scaled
        )

        pca_columns = [
            f"PCA_{i+1}"
            for i in range(
                X_pca.shape[1]
            )
        ]

        df = pd.DataFrame(
            X_pca,
            columns=pca_columns
        )

        df.insert(
            0,
            "Image_Path",
            successful_paths
        )

        if labels is not None:

            if len(labels) != len(image_paths):

                raise ValueError(
                    "Number of labels must match "
                    "number of images."
                )

            image_to_label = {
                path: labels[i]
                for i, path
                in enumerate(image_paths)
            }

            successful_labels = [
                image_to_label[path]
                for path
                in successful_paths
            ]

            df["Label"] = successful_labels

        if output_csv is not None:

            output_csv = Path(
                output_csv
            )

            output_csv.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            df.to_csv(
                output_csv,
                index=False
            )

            print(
                f"\nSaved feature dataset to:"
                f"\n{output_csv}"
            )

        print(
            "\n=============================================="
        )

        print(
            "       RadCNNFuse EXTRACTION COMPLETE"
        )

        print(
            "=============================================="
        )

        print(
            "Total images found:",
            len(image_paths)
        )

        print(
            "Successfully processed:",
            len(successful_paths)
        )

        print(
            "Failed:",
            len(failed_paths)
        )

        print(
            "Radiomics features:",
            self.EXPECTED_RADIOMICS_FEATURES
        )

        print(
            "CNN features:",
            self.EXPECTED_CNN_FEATURES
        )

        print(
            "Raw fused features:",
            X.shape[1]
        )

        print(
            "PCA components:",
            X_pca.shape[1]
        )

        print(
            "Final dataset shape:",
            df.shape
        )

        print(
            "=============================================="
        )

        return df
