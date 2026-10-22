Computer CUDA version: 13.0 

But PASCAL GPUs not supported by base cuda 13.0 pytorch version anymore. 
Uninstalled pytorch and torchvision for 13.0 

Retrying with 12.6 version. 
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

-> this worked. We proceed with this version and will use lighting. MMDetector makes more sense if we benchmark a lot, but we won't benchmark a lot as we just need a count and this computer is a brick. 


moved preprocessing data.py into data/preprocessing.py. Has not been debugged in this version, but config paths have been updated. May introduce some path errors.  Test for RF2026 data 


train sample 11.png from eval: has two pavillions marked as rented but they are white. -> fixed 


We will not infer for zone_Car. It is simply troo much out of distribution for the rest of the festival site. Count these zones manually, they are quick as well. 

later:
plot mAP evolution as function of time. Need tune learning rate. 
implement per sample metrics (maybe unecessary)
active learning pipeline


#TODO 

#### Analysis
- [ ] community, regular, special - roskilde festival camping map 
- [ ] per areal 
- [ ] god ide: dato for billede - se om du kan finde  
- [ ] lejede pavillioner til rådighed 


hvor mange skal de bruge? 
