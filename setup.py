import os
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='DigiPathAI',  
     version='0.1.1',
     author="Avinash Kori, Haran Rajkumar",
     author_email="koriavinash1@gmail.com, haranrajkumar97@gmail.com",
     description="Deep Learning toolbox for WSI (digital histopatology) analysis",
     long_description=open("README.md").read(),
     long_description_content_type="text/markdown",
     url="https://github.com/haranrk/DigiPathAI",
     packages=setuptools.find_packages(),
     package_data={'': ['LICENSE', '*/static/*','*/templates/*' ]},
     install_requires = [
         'flask',
         'openslide-python',
         ],
     extras_require={
        'gpu': [
             'torch',
             'torchvision',
             'opencv-python',
             'imgaug',
             'tqdm',
             'matplotlib',
             'scikit-learn',
             'scikit-image',
             'tensorflow-gpu>=1.14,<2',
             'pydensecrf',
             'pandas',
             'wget'],
     },
     entry_points={
         'console_scripts':['digipathai=DigiPathAI.main_server:main']
     },
     classifiers=[
         "Programming Language :: Python :: 3.5",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     include_package_data=True,
     zip_safe=False,
 )
