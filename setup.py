from setuptools import setup, find_packages

with open('README.md', "r") as f:
    readme = f.read()

with open('LICENSE') as f:
    license_content = f.read()

setup(
    name='cDeid',
    version='0.1.0',
    author='Leibo Liu',
    author_email='liuleibo@gmail.com',
    description='A Customized De-identification framework',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='',
    keywords=['DE-IDENTIFICATION', 'NLP'],
    license=license_content,
    dependency_links=['https://download.pytorch.org/whl/torch_stable.html'],
    install_requires=[
        'torch==1.6.0',
        'torchvision==0.7.0',
        'spaCy>=2.3.2',
        'stanza>=1.1.1',
        'flair==0.4.5',
        'https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.3.1/en_core_web_lg-2.3.1.tar.gz#egg=en_core_web_lg'
    ],
    packages=find_packages(exclude=('tests', 'docs')),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: Apache License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
