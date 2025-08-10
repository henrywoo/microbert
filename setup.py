from setuptools import setup
import os

setup(
    name='microbert',
    version='0.0.2',
    py_modules=['microbert.train', 'microbert.data.prepare_imdb_json'],
    packages=['microbert'],
    install_requires=[
        'torch',
        'transformers',
        'datasets',
        'numpy',
        'tqdm'
    ],
    python_requires='>=3.7',

    author='Fuheng Wu',
    author_email='wufuheng@gmail.com',
    description='A lightweight BERT implementation for text classification',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/henrywoo/microbert',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
