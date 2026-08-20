from setuptools import setup, find_packages

setup(
    name="radcnnfuse",
    version="0.1.0",
    description="Radiomics + CNN Feature Fusion Framework for Medical Image Analysis",
    author="Jerin Sultana",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "opencv-python",
        "SimpleITK",
        "pyradiomics",
        "tensorflow",
        "scikit-learn",
    ],
)
