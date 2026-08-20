import os
import numpy as np
import pandas as pd

from pathlib import Path

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
        use_radiomics=True,
        use_cnn=True
    ):

        self.image_size = image_size

        self.use_radiomics = use_radiomics
        self.use_cnn = use_cnn

        # Create engines once
        if self.use_radiomics:
            self.radiomics_extractor = (
                create_radiomics_extractor()
            )
        else:
            self.radiomics_extractor = None

        if self.use_cnn:
            self.cnn_model = (
                create_cnn_feature_model(
                    image_size=self.image_size
                )
            )
        else:
            self.cnn_model = None

    # --------------------------------------------------------
    # Extract features from one image
    # --------------------------------------------------------

    def extract_features(
        self,
        image_path
    ):

        all_features = []

        radiomics_features = None
        cnn_features = None
        mask_source = None

        # ----------------------------------------------------
        # Radiomics
        # ----------------------------------------------------

        if self.use_radiomics:

            radiomics_features, mask_source = (
                extract_radiomics_features(
                    image_path=image_path,
                    extractor=self.radiomics_extractor,
                    image_size=self.image_size
                )
            )

            all_features.append(
                radiomics_features
            )

        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        if self.use_cnn:

            cnn_features = (
                extract_cnn_features(
                    image_path=image_path,
                    model=self.cnn_model,
                    image_size=self.image_size
                )
            )

            all_features.append(
                cnn_features
            )

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        if not all_features:

            raise ValueError(
                "At least one feature extractor "
                "must be enabled."
            )

        fused_features = np.concatenate(
            all_features
        )

        return {
            "radiomics": radiomics_features,
            "cnn": cnn_features,
            "fused": fused_features,
            "mask_source": mask_source
        }

    # --------------------------------------------------------
    # Transform dataset
    # --------------------------------------------------------

    def transform_dataset(
        self,
        image_folder,
        labels=None,
        output_csv=None
    ):

        image_folder = Path(
            image_folder
        )

        if not image_folder.exists():

            raise FileNotFoundError(
                f"Image folder not found: "
                f"{image_folder}"
            )

        # ----------------------------------------------------
        # Find images
        # ----------------------------------------------------

        extensions = [
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.bmp",
            "*.tif",
            "*.tiff"
        ]

        image_paths = []

        for extension in extensions:

            image_paths.extend(
                image_folder.rglob(
                    extension
                )
            )

        image_paths = sorted(
            image_paths
        )

        if len(image_paths) == 0:

            raise ValueError(
                "No medical images found "
                f"in {image_folder}"
            )

        print(
            f"Found {len(image_paths)} images."
        )

        # ----------------------------------------------------
        # Process images
        # ----------------------------------------------------

        feature_rows = []

        successful_paths = []
        failed_paths = []

        for index, image_path in enumerate(
            image_paths,
            start=1
        ):

            try:

                result = self.extract_features(
                    str(image_path)
                )

                feature_rows.append(
                    result["fused"]
                )

                successful_paths.append(
                    str(image_path)
                )

                print(
                    f"[{index}/{len(image_paths)}] "
                    f"{image_path.name} ✓"
                )

            except Exception as e:

                failed_paths.append({
                    "path": str(image_path),
                    "error": str(e)
                })

                print(
                    f"[{index}/{len(image_paths)}] "
                    f"{image_path.name} ✗ "
                    f"{e}"
                )

        # ----------------------------------------------------
        # Check successful extraction
        # ----------------------------------------------------

        if len(feature_rows) == 0:

            raise RuntimeError(
                "No images were successfully processed."
            )

        # ----------------------------------------------------
        # Create feature matrix
        # ----------------------------------------------------

        X = np.asarray(
            feature_rows,
            dtype=np.float32
        )

        print(
            "\nFeature matrix shape:",
            X.shape
        )

        # ----------------------------------------------------
        # Create column names
        # ----------------------------------------------------

        feature_columns = [
            f"Feature_{i+1}"
            for i in range(
                X.shape[1]
            )
        ]

        df = pd.DataFrame(
            X,
            columns=feature_columns
        )

        # ----------------------------------------------------
        # Add image paths
        # ----------------------------------------------------

        df.insert(
            0,
            "Image_Path",
            successful_paths
        )

        # ----------------------------------------------------
        # Add labels if provided
        # ----------------------------------------------------

        if labels is not None:

            if len(labels) != len(image_paths):

                raise ValueError(
                    "Number of labels must match "
                    "number of images."
                )

            # Only successful images
            successful_labels = []

            path_to_label = {
                str(path): label
                for path, label
                in zip(
                    image_paths,
                    labels
                )
            }

            for path in successful_paths:

                successful_labels.append(
                    path_to_label[path]
                )

            df["Label"] = (
                successful_labels
            )

        # ----------------------------------------------------
        # Save CSV
        # ----------------------------------------------------

        if output_csv is not None:

            df.to_csv(
                output_csv,
                index=False
            )

            print(
                f"\nSaved dataset: "
                f"{output_csv}"
            )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print(
            "\n======================================"
        )

        print(
            "       RadCNNFuse Summary"
        )

        print(
            "======================================"
        )

        print(
            "Total images:",
            len(image_paths)
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
            "Feature dimension:",
            X.shape[1]
        )

        print(
            "Dataset shape:",
            df.shape
        )

        if failed_paths:

            print(
                "\nFailed images:"
            )

            for item in failed_paths:

                print(
                    item["path"],
                    "->",
                    item["error"]
                )

        return df
```
