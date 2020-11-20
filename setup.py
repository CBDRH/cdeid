from setuptools import setup, find_packages

with open('README.md', "r") as f:
    readme = f.read()

with open('LICENSE') as f:
    license_content = f.read()

setup(
    name='cdeid',
    version='0.1.2',
    author='Leibo Liu',
    author_email='liuleibo@gmail.com',
    description='A Customized De-identification framework',
    long_description_content_type='text/markdown',
    long_description=readme,
    url='https://github.com/CBDRH/cdeid',
    keywords=['DE-IDENTIFICATION', 'NLP'],
    install_requires=[
        'spaCy>=2.3.2',
        'stanza>=1.1.1',
        'flair==0.4.5',
        'mako>=1.1.3'
    ],
    packages=find_packages(exclude=('tests', 'docs')),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7'
)
