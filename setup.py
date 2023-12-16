import os
from setuptools import setup, find_packages

version = "1.0.0"

if os.getenv("RUNNING_ON_COLAB") == "1":
    setup(
        name='sam',
        version=version,
        packages=find_packages(),
    )
else:
    setup(
        name="sam",
        version=version,
        packages=["sam"],
        package_dir={"sam": "./sam"}
    )