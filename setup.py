import setuptools

__version__ = "0.0.0"

REPO_NAME = "Faultfindy-Faulty_tyre_Classification"
AUTHOR_USER_NAME = "Dhananjay"
SRC_REPO = "tyreClassifier"
AUTHOR_EMAIL = "rockingdhananjay@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for CNN app",
    long_description_content="text/markdown",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
