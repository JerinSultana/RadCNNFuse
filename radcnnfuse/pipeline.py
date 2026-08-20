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

    def __init__(
        self,
        image_size=(224, 224),
        pca_components=100
    ):
        """
        RadCNNFuse complete feature-engineering pipeline.

        Parameters
        ----------
        image_size : tuple
            Image size used for radiomics and CNN processing.

        pca_components : int
            Number of PCA components.
        """

        self.image_size = image_size
        self.pca_components = pca_components

        # --------------------------------------------------
        # Create engines
        # --------------------------------------------------

        print("Initializing RadCNNFuse...")

        self.radiomics_extractor = (
            create_radiomics_extractor()
        )

        self.cnn_feature_model = (
            create_cnn_feature_model(
                image_size=image_size
            )
        )

        # --------------------------------------------------
        # Scaler and PCA
        # --------------------------------------------------

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
        """
        Extract radiomics + CNN features from one image.
        """

        # --------------------------------------------------
        # Radiomics
        # --------------------------------------------------

        radiomics_features, mask_source = (
            extract_radiomics_features(
                image_path=image_path,
                mask_path=mask_path,
                extractor=self.radiomics_extractor,
                image_size=self.image_size
            )
        )

        # --------------------------------------------------
        # CNN
        # --------------------------------------------------

        cnn_features = extract_cnn_features(
            image_path=image_path,
            model=self.cnn_feature_model,
            image_size=self.image_size
        )

        # --------------------------------------------------
        # Fusion
        # --------------------------------------------------

        fused_features = np.concatenate(
            [
                radiomics_features,
                cnn_features
            ]
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

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
    # Dataset image discovery
    # ======================================================

    def _find_images(
        self,
        image_folder
    ):
        """
        Find supported medical images recursively.
        """

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
        """
        Complete RadCNNFuse pipeline.

        Parameters
        ----------
        image_folder : str
            Folder containing medical images.

        labels : list, optional
            Optional labels corresponding to images.

        output_csv : str, optional
            Path where PCA feature dataset will be saved.

        Returns
        -------
        pandas.DataFrame
            PCA feature dataset.
        """

        # --------------------------------------------------
        # Find images
        # --------------------------------------------------

        image_paths = self._find_images(
            image_folder
        )

        print(
            f"\nFound {len(image_paths)} images."
        )

        print(
            "\nStarting RadCNNFuse extraction..."
        )

        # --------------------------------------------------
        # Extract features
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Check extraction
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Scaling
        # --------------------------------------------------

        print(
            "\nApplying StandardScaler..."
        )

        X_scaled = self.scaler.fit_transform(
            X
        )

        # --------------------------------------------------
        # PCA
        # --------------------------------------------------

        print(
            "Applying PCA..."
        )

        # Make sure PCA components are valid
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

        # --------------------------------------------------
        # Create DataFrame
        # --------------------------------------------------

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

        # Image paths
        df.insert(
            0,
            "Image_Path",
            successful_paths
        )

        # Optional labels
        if labels is not None:

            if len(labels) != len(image_paths):

                raise ValueError(
                    "Number of labels must match "
                    "number of images."
                )

            # Keep labels corresponding to successful images
            successful_labels = []

            for path in successful_paths:

                original_index = (
                    image_paths.index(path)
                )

                successful_labels.append(
                    labels[original_index]
                )

            df["Label"] = successful_labels

        # --------------------------------------------------
        # Save CSV
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Summary
        # --------------------------------------------------

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
