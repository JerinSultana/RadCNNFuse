# RadCNNFuse

**Radiomics + CNN Feature Fusion Framework for Medical Image Analysis**

RadCNNFuse is a lightweight, research-oriented Python framework for **hybrid feature engineering from medical images**.

It combines **handcrafted radiomics features** and **deep CNN features**, fuses them, applies feature scaling and PCA, and produces a compact feature dataset that can be used with the researcher's preferred machine-learning classifier.

---

## Why RadCNNFuse?

During my undergraduate medical-image research, I found that extracting radiomics features and combining them with CNN features required **a lot of repetitive code, preprocessing steps, and time**.

RadCNNFuse was developed to make this workflow **simpler, reusable, and faster to reproduce**.

### The idea

```text
Medical Image
      ↓
   RadCNNFuse
      ↓
Radiomics + CNN Features
      ↓
     Fusion
      ↓
   Scaling + PCA
      ↓
PCA Feature Dataset
      ↓
Researcher's Classifier
```

---

## Core Pipeline

<p align="center">
  <img src="Core%20Pipeline.png" width="850" alt="RadCNNFuse Core Pipeline">
</p>

**Figure 1.** RadCNNFuse feature-engineering pipeline.

---

## Key Features

* Radiomics feature extraction
* CNN-based feature extraction
* Radiomics + CNN feature fusion
* Automatic ROI/mask support
* Feature scaling
* PCA dimensionality reduction
* Batch feature extraction
* CSV feature-dataset generation
* Reusable Python package
* Classifier-independent design

---

## Why Is It Useful for Medical-Image Researchers?

RadCNNFuse is designed to solve a common practical problem in medical-image research: **feature extraction can become repetitive and time-consuming when every experiment requires rebuilding the same pipeline.**

With RadCNNFuse, researchers can use the same feature-engineering workflow and focus on the **actual research question and classifier experimentation**.

### Researchers can benefit by:

**1. Saving development time**

Instead of repeatedly implementing radiomics extraction, CNN feature extraction, feature fusion, scaling, and PCA, researchers can use a unified pipeline.

**2. Reducing repetitive code**

The framework provides a reusable feature-engineering layer instead of requiring the complete workflow to be rewritten for every project.

**3. Separating feature engineering from classification**

Researchers can generate a PCA feature dataset once and experiment with different classifiers independently.

```text
RadCNNFuse
     ↓
PCA Features
     ↓
 ┌───────────┬───────────┬───────────┐
 │    SVM    │ Random    │    MLP    │
 │           │  Forest   │           │
 └───────────┴───────────┴───────────┘
```

**4. Making experiments easier to reproduce**

The same feature-extraction workflow can be reused across different experiments and datasets.

**5. Supporting hybrid medical-image research**

Researchers can combine handcrafted radiomics information with learned CNN representations without manually rebuilding the entire feature-fusion pipeline.

---

## Current Experimental Configuration

The current implementation was developed using kidney CT images.

| Component      | Configuration       |
| -------------- | ------------------- |
| Dataset        | Kidney CT images    |
| Images         | 745                 |
| Classes        | Cyst, Normal, Stone |
| Radiomics      | 567 features/image  |
| CNN            | MobileNetV2         |
| CNN features   | 1,280/image         |
| Fused features | 1,847/image         |
| PCA            | 100 components      |

**Note:** Feature dimensions depend on the selected radiomics settings, CNN backbone, feature-extraction layer, fusion strategy, and PCA configuration. They are not determined by the number of images.

---

## Processing Efficiency

In the current experimental environment:

```text
745 images
↓
~4.05 minutes
↓
~0.33 seconds/image
```

This timing is hardware- and configuration-dependent and should not be considered a universal benchmark.

---

## Installation

```bash
git clone https://github.com/JerinSultana/RadCNNFuse.git
cd RadCNNFuse

pip install -e .
```

If required:

```bash
pip install -r requirements.txt
```

---

## Basic Usage

```python
from radcnnfuse import RadCNNFuse

rf = RadCNNFuse()

features = rf.transform(
    "path/to/medical_image.jpg"
)
```

The resulting representation can be used for downstream machine-learning experiments.

---

## Classifier-Independent Design

RadCNNFuse does **not** force researchers to use a particular classifier.

After generating the PCA feature dataset, researchers can use algorithms such as:

* SVM
* Random Forest
* Logistic Regression
* XGBoost
* LightGBM
* CatBoost
* MLP

Example:

```python
from sklearn.svm import SVC

classifier = SVC(
    probability=True,
    random_state=42
)

classifier.fit(X_train, y_train)
```

This allows researchers to change the classifier **without rebuilding the feature-extraction pipeline**.

---

## PCA Feature Dataset

RadCNNFuse can generate a CSV dataset such as:

```text
Image_Path | PCA_1 | PCA_2 | ... | PCA_100 | Label
```

For example:

```text
745 images
×
100 PCA features
```

The generated feature dataset can be directly used in a separate machine-learning workflow.

---

## Research Workflow

```text
Medical Images
      ↓
   RadCNNFuse
      ↓
Radiomics + CNN
      ↓
Feature Fusion
      ↓
Scaling
      ↓
PCA
      ↓
Feature CSV
      ↓
Researcher's Classifier
      ↓
Evaluation
```

---

## Motivation

RadCNNFuse originated from my undergraduate thesis research in medical-image analysis.

During the research process, I experienced firsthand how time-consuming it could be to repeatedly extract radiomics features, obtain CNN representations, combine them, perform dimensionality reduction, and prepare the resulting data for machine-learning experiments.

This led to a simple idea:

> **Why rebuild the same feature-engineering pipeline for every experiment when it can be packaged into a reusable framework?**

RadCNNFuse is my attempt to turn that repetitive workflow into a reusable research tool.

---

---

## Disclaimer

RadCNNFuse is a **research and experimentation framework**.

It is not a certified medical device and should not be used for clinical diagnosis or treatment decisions.

Experimental results should not be interpreted as clinical diagnostic performance.

---

## Roadmap

* [ ] More radiomics configurations
* [ ] Additional CNN backbones
* [ ] More preprocessing options
* [ ] Additional dimensionality-reduction methods
* [ ] Automated classifier benchmarking
* [ ] Improved documentation
* [ ] PyPI release
* [ ] More medical-image format support
* [ ] Unit and integration tests

---

---

## Developed by

**Jerin Sultana**
Department of Computer Science and Engineering
University of Science and Technology Chittagong (USTC)

---

## Project Status

**Research Prototype / Experimental Release**

RadCNNFuse is currently intended for **research, experimentation, and medical-image feature engineering**.

Further benchmarking, documentation, testing, and external validation are planned.
