import os

import setuptools

__folder__ = os.path.abspath(os.path.dirname(__file__))


if __name__ in ["__main__", "builtins"]:
    setuptools.setup(
        name="ml101",
        include_package_data=True,
        packages=setuptools.find_packages(),
        python_requires=">=3.7",
        install_requires=[
            "uvloop",
            "ujson",
            "websockets",
            "bidict",
            "numpy",
            "pandas",
            "dask",
            "dask-ml",
            "scikit-learn",
            "dask_xgboost",
            "fsspec",
            "imbalanced-learn",
            "seaborn",
        ],
        entry_points={"console_scripts": ["ml101=ml101.main:main"]},
    )
