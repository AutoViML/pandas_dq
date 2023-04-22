import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pandas_dq",
    version="1.10",
    author="Ram Seshadri",
    # author_email="author@example.com",
    description="Clean your data using a scikit-learn transformer in a single line of code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    url="https://github.com/AutoViML/pandas_dq",
    py_modules = ["pandas_dq"],
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=[
        "numpy>=1.21.5",
        "pandas>=1.3.5",
        "scikit-learn>=0.24.2",
    ],
    include_package_data = True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
