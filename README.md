# RadCNNFuse

**Radiomics + CNN Feature Fusion Framework for Medical Image Analysis**

RadCNNFuse is a research-oriented Python framework designed to simplify hybrid feature engineering from medical images by combining handcrafted **radiomics features** with deep **CNN features**, followed by feature fusion, scaling, and PCA-based dimensionality reduction.

The framework was motivated by a practical problem encountered during undergraduate medical-image research: manually reproducing radiomics extraction and hybrid feature-engineering pipelines can be time-consuming, repetitive, and difficult to maintain.

RadCNNFuse aims to turn this repetitive workflow into a **reusable, lightweight, and classifier-independent feature-engineering pipeline**.

---

## Why RadCNNFuse?

Hybrid medical-image analysis commonly involves several separate stages:

1. Image preprocessing
2. ROI/mask generation
3. Radiomics feature extraction
4. Deep CNN feature extraction
5. Feature fusion
6. Feature scaling
7. Dimensionality reduction
8. Downstream classification

Rebuilding these steps manually can require substantial code and repeated experimentation.

RadCNNFuse provides a unified feature-engineering layer so researchers can focus more on their experiments and less on repeatedly implementing the same extraction workflow.

### The main idea

```text
Medical Image
      ↓
   RadCNNFuse
      ↓
PCA Feature Dataset
      ↓
Researcher's Classifier
```

RadCNNFuse handles the feature-engineering stage, while the researcher remains free to select the downstream machine-learning classifier.

---
## RadCNNFuse Core Pipeline

The proposed **RadCNNFuse** framework integrates handcrafted radiomics features with deep CNN features, followed by feature fusion, standard scaling, and PCA-based dimensionality reduction to generate a compact feature representation for downstream machine-learning experiments.

<p align="center">
  <img src="Core%20Pipeline.png" width="850" alt="RadCNNFuse Core Pipeline">
</p>

**Figure 1.** Overall architecture of the RadCNNFuse framework, including radiomics and CNN feature extraction, feature fusion, scaling, PCA-based dimensionality reduction, and downstream classifier selection.

---

## Core Pipeline

RadCNNFuse combines two complementary feature representations:

* **Handcrafted radiomics features**, which describe image intensity, texture, and spatial characteristics.
* **Deep CNN features**, which provide learned visual representations from a convolutional neural network.

These representations are concatenated into a hybrid feature vector and subsequently transformed using scaling and PCA.


## Key Features

* Automatic radiomics feature extraction
* CNN-based deep feature extraction
* Radiomics + CNN feature fusion
* Automatic ROI/mask generation
* External mask support
* Standard feature scaling
* PCA-based dimensionality reduction
* Batch feature extraction
* Checkpoint-supported processing
* CSV feature-dataset generation
* Reusable Python package interface
* Classifier-independent feature representation
* End-to-end feature extraction and transformation
* Designed for research and experimentation

---

## 📊 Current Experimental Configuration

The current implementation was developed and validated using kidney CT images.

| Component                | Current Configuration |
| ------------------------ | --------------------- |
| Dataset                  | Kidney CT images      |
| Dataset size             | 745 images            |
| Classes                  | Cyst, Normal, Stone   |
| Radiomics representation | 567 features/image    |
| CNN backbone             | MobileNetV2           |
| CNN representation       | 1,280 features/image  |
| Fused representation     | 1,847 features/image  |
| PCA representation       | 100 features/image    |
| PCA variance retained*   | 92.39%                |

*The reported 92.39% variance retention refers to the PCA representation generated for the full feature dataset for feature export.

For unbiased machine-learning evaluation, **scaling and PCA should be fitted only on the training data and then applied to the held-out test data**. This protocol was followed in the reported test-set evaluation.

### Important

The feature dimensions above are **configuration-dependent**.

They are determined by:

* selected radiomics feature classes and settings,
* enabled image types,
* CNN backbone and feature-extraction layer,
* feature-fusion strategy,
* PCA configuration.

Therefore, the dimensions are not dependent on the number of images.

For example:

```text
1 image
→ 567 radiomics features
→ 1,280 CNN features
→ 1,847 fused features
→ 100 PCA features
```

For `N` images:

```text
N × 1,847 fused feature matrix
```

and after PCA:

```text
N × 100 PCA feature matrix
```

---

## ⚡ Processing Efficiency

RadCNNFuse is designed to reduce repetitive implementation and feature-engineering overhead.

In the current experimental environment, the complete batch feature-extraction process was tested on **745 kidney CT images**.

```text
Total images              : 745
Successfully processed    : 745
Failed                    : 0
Total extraction time     : ~4.05 minutes
Average time/image        : ~0.33 seconds
```

The pipeline generated:

```text
567 radiomics features/image
+
1,280 CNN features/image
=
1,847 fused features/image
```

The reported processing time is specific to the tested hardware, preprocessing configuration, radiomics settings, CNN backbone, and implementation environment. It should therefore not be interpreted as a universal benchmark.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/JerinSultana/RadCNNFuse.git
cd RadCNNFuse
```

Install the package:

```bash
pip install -e .
```

Install dependencies if required:

```bash
pip install -r requirements.txt
```

---

## Package Interface

The intended package interface is:

```python
from radcnnfuse import RadCNNFuse

rf = RadCNNFuse()
```

After initialization, the framework can be used to transform medical images into reduced feature representations.

Example:

```python
features = rf.transform(
    "path/to/medical_image.jpg"
)
```

The exact available methods and configuration options may depend on the installed version of RadCNNFuse.

---

## Feature Extraction

RadCNNFuse independently extracts radiomics and CNN representations.

Conceptually:

```python
radiomics_features = ...
cnn_features = ...

fused_features = concatenate(
    [radiomics_features, cnn_features]
)
```

In the current experimental configuration:

```text
Radiomics → 567 features
CNN      → 1,280 features
Fusion   → 1,847 features
```

---

## PCA-Based Dimensionality Reduction

After feature fusion, RadCNNFuse applies standard scaling followed by PCA.

Current experimental configuration:

```text
Input features     : 1,847
PCA components     : 100
```

The resulting representation is:

```text
1,847-dimensional
        ↓
      PCA
        ↓
100-dimensional
```

This substantially reduces the dimensionality of the hybrid feature representation while retaining a large proportion of the variance under the tested configuration.

---

## Classifier-Independent Design

One of the main design goals of RadCNNFuse is to keep **feature extraction separate from downstream classification**.

RadCNNFuse does not require researchers to use a specific classifier.

After obtaining the PCA feature dataset, researchers can train different machine-learning models according to their experimental requirements.

### Example: SVM

```python
from sklearn.svm import SVC

classifier = SVC(
    probability=True,
    random_state=42
)

classifier.fit(
    X_train,
    y_train
)
```

### Example: Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

classifier.fit(
    X_train,
    y_train
)
```

Other compatible classifiers can also be investigated, including:

* Logistic Regression
* XGBoost
* LightGBM
* CatBoost
* Random Forest
* SVM
* MLP
* Other machine-learning classifiers

This allows researchers to experiment with different downstream models **without rebuilding the radiomics + CNN feature-extraction pipeline**.

---

## PCA Dataset Output

RadCNNFuse can generate a PCA-reduced CSV feature dataset.

A typical output contains:

```text
Image_Path
PCA_1
PCA_2
PCA_3
...
PCA_100
Label
```

For the current experimental dataset:

```text
Samples       : 745
PCA features  : 100
```

Therefore, the feature matrix is:

```text
745 × 100
```

plus the image-path and label columns.

Example:

```text
Image_Path | PCA_1 | PCA_2 | ... | PCA_100 | Label
```

The generated dataset can then be loaded into a separate machine-learning workflow.

---

## Typical Research Workflow

A researcher can use RadCNNFuse as a feature-engineering layer:

```text
Medical Images
      │
      ▼
   RadCNNFuse
      │
      ├── Radiomics
      ├── CNN
      ├── Fusion
      ├── Scaling
      └── PCA
      │
      ▼
PCA Feature CSV
      │
      ▼
Train / Test Split
      │
      ▼
Researcher's Classifier
      │
      ▼
Evaluation
```

This design allows the same extracted representation to be investigated using multiple downstream classifiers.

---

## Motivation

RadCNNFuse originated from undergraduate thesis research involving medical CT images.

During the research process, extracting handcrafted radiomics features required substantial time and repeated implementation. Combining those handcrafted representations with CNN features introduced additional preprocessing, feature management, scaling, fusion, and dimensionality-reduction steps.

The experience motivated the development of a reusable framework that could turn this multi-stage workflow into a simpler and more systematic process.

### The motivation in one sentence

> **Instead of repeatedly rebuilding the radiomics + CNN feature-engineering pipeline for every experiment, RadCNNFuse provides a reusable feature-engineering layer that produces a ready-to-use reduced feature representation.**

The long-term objective is to make hybrid feature engineering more accessible for researchers working on medical-image analysis.

---
## ⚠️ Disclaimer

RadCNNFuse is a **research and experimentation framework**.

It is not a certified medical device and should not be used for:

* clinical diagnosis,
* treatment decisions,
* patient management,
* or other clinical decision-making.

The experimental results reported in this repository should not be interpreted as evidence of clinical effectiveness.

Further external validation on independent and diverse datasets is required before any potential clinical application.

---

## Roadmap

Future development may include:

* [ ] Configurable radiomics feature classes
* [ ] Additional CNN backbones
* [ ] Multiple preprocessing strategies
* [ ] Additional dimensionality-reduction methods
* [ ] Automated classifier benchmarking
* [ ] Improved batch-processing utilities
* [ ] Improved error handling
* [ ] PyPI-ready packaging
* [ ] Comprehensive API documentation
* [ ] Unit tests
* [ ] Integration tests
* [ ] Reproducible experiment configurations
* [ ] Support for additional medical-image formats
* [ ] Optional GPU acceleration

---

## Contributing

Contributions, suggestions, bug reports, and research collaborations are welcome.

If you find a problem or have a suggestion, please open a GitHub issue.

Pull requests are also welcome.

For major methodological changes, opening an issue before submitting a pull request is recommended.

---

## 👩‍💻 Author

**Jerin Sultana**

Department of Computer Science and Engineering
University of Science and Technology Chittagong (USTC)

---

## Acknowledgment

RadCNNFuse was developed as part of undergraduate research in medical image analysis and was motivated by practical challenges encountered during radiomics-based feature extraction and hybrid deep-learning experimentation.

The project represents an effort to transform a repetitive research workflow into a reusable feature-engineering framework.

If you find RadCNNFuse useful for your research, consider giving the repository a ⭐ on GitHub.

---

## Project Status

**Current status: Research Prototype / Experimental Release**

The current version has been experimentally validated for the kidney CT image classification workflow described in this repository.

The framework is intended primarily for **research, experimentation, and feature-engineering studies**. Further testing, documentation, benchmarking, and external validation are planned for future releases.
