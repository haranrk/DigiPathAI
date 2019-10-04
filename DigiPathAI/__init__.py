from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from time import gmtime, strftime
from google_drive_downloader import GoogleDriveDownloader as gdd

from .helpers import *
from .models import *
from .loaders import *

model_path = os.path.join('../model_weights')
if (not os.path.exists(model_path)) or (os.listdir(model_path) == []):
	os.makedirs(model_path)
	print ("[INFO: DigiPathAI] (" + strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + ") " + 'Densenet1')
	gdd.download_file_from_google_drive(file_id='1uo5RcjJ-QBXUgtaDfvqEx9JJAbS5H-Ja',
                                    dest_path=os.path.join(model_path, 'densenet_fold1.h5'))

	print ("[INFO: DigiPathAI] (" + strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + ") " + 'Densenet2')
	gdd.download_file_from_google_drive(file_id='1vA-r2hPcmhmwm92j9VCeeDOIBRI5IowV',
                                    dest_path=os.path.join(model_path, 'densenet_fold2.h5'))

	print ("[INFO: DigiPathAI] (" + strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + ") " + 'Inception ')
	gdd.download_file_from_google_drive(file_id='1AyU7m-Gsnnsva77Ie1-eUkmI1Rijyfhy',
                                    dest_path=os.path.join(model_path, 'inception.h5'))

	print ("[INFO: DigiPathAI] (" + strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + ") " + 'DeepLabv3')
	gdd.download_file_from_google_drive(file_id='1bv-kN2WuaYvkgPLZx5ACcq2qRab5OiD2',
                                    dest_path=os.path.join(model_path, 'deeplabv3.h5'))
else :
	print ("[INFO: DigiPathAI] (" + strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + ") " + 'Skipping Download Files already exists')

