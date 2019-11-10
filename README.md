# DigiPathAI
A software application built on top of [openslide](https://openslide.org/) for viewing [whole slide images (WSI)](https://www.ncbi.nlm.nih.gov/pubmed/30307746) and performing pathological analysis 

# Features
- Responsive WSI image viewer 
- State of the art cancer AI pipeline to segment the and display the cancer cell

# Online Demo
https://digipathai.tech/

# Installation
Running of the AI pipeline requires a GPU and several deep learning modules. However, you can run just the UI as well.

## Just the UI
### Requirements
- `openslide`
- `flask`

The following command will install only the dependencies listed above.
```
pip install DigiPathAI
```

## Entire AI pipeline
### Requirements
- `pytorch`
- `torchvision`
- `opencv-python`
- `imgaug`
- `matplotlib`
- `scikit-learn`
- `scikit-image`
- `tensorflow-gpu >=1.14,<2`
- `pydensecrf`
- `pandas`
- `wget`

The following command will install the dependencies mentioned
```
pip install "DigiPathAI[gpu]"
```

Both installation methods install the same package, just different dependencies. Even if you had installed using the earlier command, you can install the rest of the dependencies manually. 


# Usage 
## Local server
Traverse to the directory containing the openslide images and run the following command.
```
digipathai <host: localhost (default)> <port: 8080 (default)>
```

## Python API usage
The application also has an API which can be used within python to perform the segmentation. 
```
from DigiPathAI.Segmentation import getSegmentation

prediction = getSegmentation(img_path, 
			patch_size  = 256, 
			stride_size = 128,
			batch_size  = 32,
			quick       = True,
			tta_list    = None,
			crf         = False,
			save_path   = None,
			status      = None)
```

# Contact
- Avinash Kori (koriavinash1@gmail.com)
- Haran Rajkumar (haranrajkumar97@gmail.com)

