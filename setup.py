from setuptools import setup, find_packages

setup(
    name='scorch',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas', 'numpy', 'torch'
    ],
    author='Ross Heaton',
    author_email='rossalanheaton@gmail.com',
    description="Implementation of Pytorch for educational purposes.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    python_requires='>=3.9',
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)