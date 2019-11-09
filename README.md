# DigiPathAI
A software application for viewing [whole slide images (WSI)](https://www.ncbi.nlm.nih.gov/pubmed/30307746) and performing pathological analysis 

# Features
- Responsive WSI image viewer 
- State of the art cancer AI pipeline to segment the and display the cancer cell

# Installation
```
pip install DigiPathAI
```

# Usage Local server
```
digipathai <host: localhost (default)> <port: 8080 (default)>
```

# Python API usage
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

# Results
![results](./imgs/results_1.png')

# Application Overview
![demo](./imgs/demo.gif')

# Requirements
Just for the viewer 
- `openslide`

For the segmentation as well
- `openslide`
- `tensorflow<2.0.0`
- `pytorch`

# Contact
- Avinash Kori (koriavinash1@gmail.com)
- Haran Rajkumar (haranrajkumar97@gmail.com)

