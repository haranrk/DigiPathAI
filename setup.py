import os
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='DigiPathAI',  
     version='0.0.1',
     author="Avinash Kori, Haran Rajkumar",
     author_email="koriavinash1@gmail.com, haranrajkumar97@gmail.com",
     description="Deep Learning toolbox for WSI (digital histopatology) analysis",
     long_description=open("README.md").read(),
     long_description_content_type="text/markdown",
     url="https://github.com/haranrk/DigiPathAI",
     packages=setuptools.find_packages(),
     install_requires = [
         'torch',
         'torchvision',
         'opencv-python',
         'imgaug',
         'tqdm',
         'matplotlib',
         'scikit-learn',
         'scikit-image',
         'tensorflow-gpu==1.14',
         'flask',
         'pydensecrf',
         'openslide-python',
         'pandas',
         'wget'
         ],
     entry_points='''
        [console_scripts]
        digipathai=DigiPathAI.main_server:main
     ''',
     classifiers=[
         "Programming Language :: Python :: 3.5",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
