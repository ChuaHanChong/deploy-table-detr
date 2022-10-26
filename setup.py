from setuptools import find_namespace_packages, setup

setup(
    name="table_detr",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
)
