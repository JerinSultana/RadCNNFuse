# ============================================================
# RadCNNFuse V2
# pipeline.py
# ============================================================

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

    # ========================================================
    # Initialization
    # ========================================================

    def __init__(
        self,
        image_size=(150, 150),
        pca_components=100,
        valid_classes=None
    ):
        """
        RadCNNFuse feature-fusion pipeline.

        Pipeline:

            Medical Image
                    ↓
              ROI / Mask
                    ↓
              Radiomics
                    ↓
             CNN Features
                    ↓
                 Fusion
                    ↓
             StandardScaler
                    ↓
                  PCA

        Parameters
        ----------
        image_size : tuple
            Image size used by the pipeline.

        pca_components : int
            Number of PCA components.

        valid_classes : set/list
            Valid medical-image class folders.
        """

        self.image_size = image_size
        self.pca_components = pca_components

        # ----------------------------------------------------
        # Valid classes
        # ----------------------------------------------------

        if valid_classes is None:

            valid_classes = {
                "Cyst",
                "Normal",
                "Stone"
            }

        self.valid_classes = set(
            valid_classes
        )

        print(
            "Initializing RadCNNFuse..."
        )

        # ----------------------------------------------------
        # Radiomics engine
        # ----------------------------------------------------

        self.radiomics_extractor = (
            create_radiomics_extractor()
        )

        # ----------------------------------------------------
        # CNN engine
        # ----------------------------------------------------

        self.cnn_feature_model = (
            create_cnn_feature_model(
                image_size=image_size
            )
        )

        # ----------------------------------------------------
        # Scaler
        # ----------------------------------------------------

        self.scaler = StandardScaler()

        # ----------------------------------------------------
        # PCA
        # ----------------------------------------------------

        self.pca = PCA(
            n_components=pca_components,
            random_state=42
        )

        print(
            "Radiomics engine: READY"
        )

        print(
            "CNN engine: READY"
        )

        print(
            "Scaler: READY"
        )

        print(
            "PCA: READY"
        )

        print(
            f"Image size: {image_size}"
        )

        print(
            f"Valid classes: "
            f"{sorted(self.valid_classes)}"
        )

    # ========================================================
    # Single Image Feature Extraction
    # ========================================================

    def extract_features(
        self,
        image_path,
        mask_path=None
    ):
        """
        Extract radiomics + CNN features
        from a single medical image.
        """

        # ----------------------------------------------------
        # Radiomics
        # ----------------------------------------------------

        (
            radiomics_features,
            mask_source
        ) = extract_radiomics_features(

            image_path=image_path,

            mask_path=mask_path,

            extractor=self.radiomics_extractor,

            image_size=self.image_size
        )

        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        cnn_features = extract_cnn_features(

            image_path=image_path,

            model=self.cnn_feature_model,

            image_size=self.image_size
        )

        # ----------------------------------------------------
        # Convert to 1D arrays
        # ----------------------------------------------------

        radiomics_features = np.asarray(
            radiomics_features,
            dtype=np.float32
        ).reshape(-1)

        cnn_features = np.asarray(
            cnn_features,
            dtype=np.float32
        ).reshape(-1)

        # ----------------------------------------------------
        # Feature Fusion
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
                "NaN or Inf detected "
                "in extracted features."
            )

        return {

            "radiomics": radiomics_features,

            "cnn": cnn_features,

            "fused": fused_features,

            "mask_source": mask_source
        }

    # ========================================================
    # Dataset Discovery
    # ========================================================

    def _find_images(
        self,
        image_folder
    ):
        """
        Find only valid medical images.

        Only images whose immediate parent folder is:

            Cyst
            Normal
            Stone

        are included.

        This prevents unrelated mask/output folders
        from entering the RadCNNFuse pipeline.
        """

        image_folder = Path(
            image_folder
        )

        if not image_folder.exists():

            raise FileNotFoundError(
                f"Image folder not found:\n"
                f"{image_folder}"
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

        # ----------------------------------------------------
        # Recursive search
        # ----------------------------------------------------

        for path in image_folder.rglob("*"):

            if not path.is_file():
                continue

            if path.suffix.lower() not in extensions:
                continue

            # ------------------------------------------------
            # Immediate parent folder
            # ------------------------------------------------

            class_name = path.parent.name

            # ------------------------------------------------
            # Keep only valid classes
            # ------------------------------------------------

            if class_name not in self.valid_classes:
                continue

            image_paths.append(
                str(path)
            )

        # ----------------------------------------------------
        # Sort
        # ----------------------------------------------------

        image_paths.sort()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if len(image_paths) == 0:

            raise ValueError(
                "No valid medical images found.\n"
                f"Expected class folders: "
                f"{sorted(self.valid_classes)}"
            )

        return image_paths

    # ========================================================
    # Get Labels Automatically
    # ========================================================

    def _get_labels_from_paths(
        self,
        image_paths
    ):
        """
        Automatically obtain class labels
        from the immediate parent folder.
        """

        labels = []

        for path in image_paths:

            class_name = (
                Path(path).parent.name
            )

            if class_name not in self.valid_classes:

                raise ValueError(
                    f"Invalid class detected: "
                    f"{class_name}"
                )

            labels.append(
                class_name
            )

        return labels

    # ========================================================
    # Full Dataset Transformation
    # ========================================================

    def transform_dataset(
        self,
        image_folder,
        labels=None,
        output_csv=None
    ):
        """
        Complete RadCNNFuse dataset pipeline.

        Steps:

            1. Discover valid CT images
            2. Extract radiomics
            3. Extract CNN features
            4. Fuse features
            5. StandardScaler
            6. PCA
            7. Create final DataFrame
            8. Optionally save CSV

        Parameters
        ----------
        image_folder : str
            Main dataset folder.

        labels : list, optional
            Optional external labels.

            If None, labels are automatically obtained
            from Cyst / Normal / Stone folders.

        output_csv : str, optional
            Output CSV path.

        Returns
        -------
        pandas.DataFrame
        """

        # ====================================================
        # 1. Find valid images
        # ====================================================

        image_paths = self._find_images(
            image_folder
        )

        print(
            "\n=============================================="
        )

        print(
            "       RadCNNFuse Dataset Discovery"
        )

        print(
            "=============================================="
        )

        print(
            "Valid CT images found:",
            len(image_paths)
        )

        # ====================================================
        # 2. Generate labels
        # ====================================================

        if labels is None:

            labels = self._get_labels_from_paths(
                image_paths
            )

        else:

            if len(labels) != len(image_paths):

                raise ValueError(
                    "Number of labels must match "
                    "number of discovered images."
                )

            labels = list(labels)

        # ====================================================
        # 3. Class distribution
        # ====================================================

        print(
            "\nClass distribution:"
        )

        class_counts = (
            pd.Series(labels)
            .value_counts()
        )

        for class_name in [
            "Cyst",
            "Normal",
            "Stone"
        ]:

            print(
                f"{class_name}: "
                f"{class_counts.get(class_name, 0)}"
            )

        # ====================================================
        # 4. Feature extraction
        # ====================================================

        feature_rows = []

        successful_paths = []

        successful_labels = []

        failed_paths = []

        print(
            "\n=============================================="
        )

        print(
            "       RadCNNFuse Extraction"
        )

        print(
            "=============================================="
        )

        start_time = __import__(
            "time"
        ).time()

        total_images = len(
            image_paths
        )

        for index, image_path in enumerate(
            image_paths,
            start=1
        ):

            try:

                result = self.extract_features(
                    image_path=image_path
                )

                fused = result[
                    "fused"
                ]

                feature_rows.append(
                    fused
                )

                successful_paths.append(
                    image_path
                )

                # --------------------------------------------
                # Corresponding label
                # --------------------------------------------

                image_index = index - 1

                successful_labels.append(
                    labels[image_index]
                )

                print(
                    f"[{index}/{total_images}] "
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
                    f"[{index}/{total_images}] "
                    f"FAILED: "
                    f"{Path(image_path).name}"
                )

                print(
                    f"    Error: {e}"
                )

        elapsed = (
            __import__("time").time()
            - start_time
        )

        # ====================================================
        # 5. Extraction validation
        # ====================================================

        if len(feature_rows) == 0:

            raise RuntimeError(
                "No images were successfully processed."
            )

        X = np.asarray(
            feature_rows,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Check feature dimension
        # ----------------------------------------------------

        expected_radiomics = 567
        expected_cnn = 1280
        expected_fused = (
            expected_radiomics
            + expected_cnn
        )

        print(
            "\n=============================================="
        )

        print(
            "       Extraction Summary"
        )

        print(
            "=============================================="
        )

        print(
            "Total images:",
            total_images
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
            "Raw fused feature matrix:",
            X.shape
        )

        print(
            "Expected fused features:",
            expected_fused
        )

        print(
            "Processing time:",
            f"{elapsed / 60:.2f} minutes"
        )

        if X.shape[1] != expected_fused:

            raise ValueError(
                f"Unexpected fused feature dimension: "
                f"{X.shape[1]}. "
                f"Expected {expected_fused} "
                f"(567 radiomics + 1280 CNN)."
            )

        # ====================================================
        # 6. StandardScaler
        # ====================================================

        print(
            "\nApplying StandardScaler..."
        )

        X_scaled = self.scaler.fit_transform(
            X
        )

        # ====================================================
        # 7. PCA
        # ====================================================

        print(
            "Applying PCA..."
        )

        n_components = min(
            self.pca_components,
            X_scaled.shape[0],
            X_scaled.shape[1]
        )

        if (
            n_components
            != self.pca_components
        ):

            self.pca = PCA(
                n_components=n_components,
                random_state=42
            )

        X_pca = self.pca.fit_transform(
            X_scaled
        )

        # ====================================================
        # 8. PCA variance
        # ====================================================

        explained_variance = (
            self.pca
            .explained_variance_ratio_
            .sum()
        )

        print(
            "PCA components:",
            X_pca.shape[1]
        )

        print(
            "Variance retained:",
            f"{explained_variance * 100:.2f}%"
        )

        # ====================================================
        # 9. Create DataFrame
        # ====================================================

        pca_columns = [

            f"PCA_{i + 1}"

            for i in range(
                X_pca.shape[1]
            )
        ]

        df = pd.DataFrame(
            X_pca,
            columns=pca_columns
        )

        # ----------------------------------------------------
        # Add label
        # ----------------------------------------------------

        df["Label"] = (
            successful_labels
        )

        # ----------------------------------------------------
        # Image path
        # ----------------------------------------------------

        df.insert(
            0,
            "Image_Path",
            successful_paths
        )

        # ====================================================
        # 10. Save CSV
        # ====================================================

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
                "\nSaved feature dataset to:"
            )

            print(
                output_csv
            )

        # ====================================================
        # 11. Save failed images
        # ====================================================

        if len(failed_paths) > 0:

            failed_df = pd.DataFrame(
                failed_paths
            )

            failed_csv = Path(
                output_csv
                if output_csv is not None
                else "RadCNNFuse_failed_images.csv"
            )

            failed_csv = failed_csv.with_name(
                failed_csv.stem
                + "_FAILED.csv"
            )

            failed_df.to_csv(
                failed_csv,
                index=False
            )

            print(
                "\nFailed-image report saved to:"
            )

            print(
                failed_csv
            )

        # ====================================================
        # 12. Final summary
        # ====================================================

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
            "Radiomics features:",
            expected_radiomics
        )

        print(
            "CNN features:",
            expected_cnn
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
