import codecs
import os
import re
from setuptools import find_packages
from setuptools import setup


def readme():
    with codecs.open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "version.py")
    version_regex = r"__version__ = ['\"]([^'\"]*)['\"]"
    with open(version_file, "r") as f:
        version = re.search(version_regex, f.read(), re.M).group(1)
        return version


def parse_requirements(fname='requirements.txt'):
    """Parse the package dependencies listed in a requirements file."""

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for line in parse_require_file(target):
                yield line
        else:
            yield line

    def parse_require_file(fpath):
        with codecs.open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for ll in parse_line(line):
                        yield ll

    packages = list(parse_require_file(fname))
    return packages


setup(
    name='easy_tpp',
    version=get_version(),
    description='An easy and flexible tool for neural temporal point process',
    long_description=readme(),
    author='Alipay',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3'
    ],
    install_requires=parse_requirements('requirements.txt'))
