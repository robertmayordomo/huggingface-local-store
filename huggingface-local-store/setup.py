from setuptools import setup, find_packages

setup(
    name='huggingface-local-store',
    version='0.0.1',
    author='Robert Butler',
    author_email='robert@mayordomo.co.uk',
    description='Copy Huggingface models to the cloud',
    long_description='Enable Copying Of Hugging face models to Azure Cloud.',
    url='https://github.com/robertmayordomo/huggingface-local-store',
    packages=find_packages(),
    install_requires=[
        'transformers>=4.30.2',
        'azure-storage-blob>=12.18.1',
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',        
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
