from setuptools import setup

def parse_requirements(filename):
    with open(filename) as f:
        return f.readlines()

setup(
    name='mnistified',
    packages=['mnistified'],
    include_package_data=True,
    install_requires=parse_requirements('requirements.txt'),
    test_requires=parse_requirements('requirements_test.txt'),
)