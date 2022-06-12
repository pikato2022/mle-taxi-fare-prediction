import setuptools

REQUIRED_PACKAGES = [
    "google-cloud-aiplatform==1.13.2",
]

setuptools.setup(
    name="executor",
    version="0.0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    # include_package_data=True,
    # package_data={"src": ["raw_schema/schema.pbtxt"]},
)
