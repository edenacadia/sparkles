#stolen from joseph

from setuptools import setup, find_packages
from os import path

HERE = path.abspath(path.dirname(__file__))
PROJECT = 'sparkles'

with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

with open(path.join(HERE, PROJECT, 'VERSION'), encoding='utf-8') as f:
    VERSION = f.read().strip()

setup(
    name=PROJECT,
    version=VERSION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.9, <4',
    install_requires=[],
    package_data={  # Optional
        PROJECT: ['VERSION'],
    },
    project_urls={  # Optional
        'Bug Reports': f'https://github.com/edenacadia/{PROJECT}/issues',
    },
)