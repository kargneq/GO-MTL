from setuptools import setup, find_packages

setup(
    name='go_mtl',
    version='0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
    ],
)