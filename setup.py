import setuptools

setuptools.setup(
        name='geebeam',
        version='0.1.0',
        install_requires=[
"apache-beam[gcp]>=2.70.0",
"dill>=0.4.1",
"earthengine-api>=1.7.10",
"geopandas>=1.0.1",
"tensorflow>=2.16",
"tensorflow-data-validation>=1.16",
"tensorflow-metadata>=1.16",
"tfx-bsl>=1.16",
        ],
)
