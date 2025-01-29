# setup.py

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="crawlmind",  # Your package name on PyPI
    version="0.1.0",
    author="Ashish Kumar Singh",
    author_email="ashishkmr472@gmail.com",
    description="A LLM-powered OSINT web crawler package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ashishkumar4/crawlmind",
    packages=setuptools.find_packages(exclude=["tests*", "examples*"]),
    include_package_data=True,
    install_requires=[
        "openai>=0.28.0",
        "pydantic>=1.10.0",
        "requests>=2.0.0",
        "playwright>=1.30.0",
        "beautifulsoup4>=4.0.0",
        "colorlog>=6.0.0",
        "Flask>=2.0.0",
    ],
    python_requires=">=3.9",
    license="MIT",  # or your chosen license
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "crawlmind-server=server:app.run",
        ],
    },
)
