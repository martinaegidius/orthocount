Computer CUDA version: 13.0 

But PASCAL GPUs not supported by base cuda 13.0 pytorch version anymore. 
Uninstalled pytorch and torchvision for 13.0 

Retrying with 12.6 version. 
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

-> this worked. We proceed with this version and will use lighting. MMDetector makes more sense if we benchmark a lot, but we won't benchmark a lot as we just need a count and this computer is a brick. 


moved preprocessing data.py into data/preprocessing.py. Has not been debugged in this version, but config paths have been updated. May introduce some path errors.  Test for RF2026 data 

