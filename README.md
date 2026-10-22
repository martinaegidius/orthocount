# orthocount

End-to-end geospatial computer vision pipeline for counting festival tents and pavilions from aerial orthophotos.

## Installation 
Create an environment. 
Run `pip install -e .`

This install requirements.txt and orthocount as a package. 

In addition, install the fitting version of torch for your computer, e.g.:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`

If you want to make small adaptations of the code only, reinstall the changed modules using: 
`pip install -e . --no-deps`

## Running 
The pipeline can be run as-is for inference on your own pavillion dataset, or can be used for preprocessing, training a model for another goal, and analysis. 

#### GDAL Preprocessing 
Add your data to the *data/* folder, and adapt the *gdal_preprocessing* and *storage* configs accordingly. The main configuration is *config.yaml*. Use this as a point of reference for getting started. E.g. assume you have a geotiff dataset for 2026: 
1. Create a folder and place the dataset in *data/2026/input_resolution/geotiff/yourgeo.tiff*
2. Change the **year** and **src_geotiff_filename** in the preprocessing-configuration you use to point to the new file 
3. Ensure that *config.yaml* points to the storage option you use. For external storage, use the *storage/hdd.yaml* configuration, and create a path for the dataset here. 

Now run: 
``python
python3 -m src.orthocount.data.preprocess
``

The output will be created in *data* as according to gdal_preprocessing config of choice. 


#### Tiling into an image dataset 
Fill out as you go for 2026

## TODO 

- Update requirements.txt 
- Add __init__.py files everywhere..

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## Decision logs 

### Resolution
We tune the canonical pixel resolution to support potential downstream tent-detection. As a result, we use the following [table values](https://www.harald-nyborg.dk/harald-nyborg-festivaltelt-2-personer) for the smallest commonly occuring festival tent: 
Width: 1,4 m
Length: 2,05 m

In order to safely be able to detect this size, we will want it to be at least 20 pixels in the shortest dimension. As a result we need $140cm/20pixels = 7 cm/pixel$ resolution. In order to allow potential other users to use the models, the target resolution thus is chosen as 10 cm per pixel unless this proves impractical. 

We assume white pavillions to be the common harald nyborg 3x3 pavillion: 
https://www.harald-nyborg.dk/pavillon-3-x-3-meter-hvid?qwerty=Cj0KCQjwkOvTBhDgARIsAKUNyRuHnUk1V0mioF32qBSkxz2GonGyDYgHOWZOe4IGIFyicT4kmbwPzNMaAgEbEALw_wcB&gad_source=1&gad_campaignid=22306451493&gclid=Cj0KCQjwkOvTBhDgARIsAKUNyRuHnUk1V0mioF32qBSkxz2GonGyDYgHOWZOe4IGIFyicT4kmbwPzNMaAgEbEALw_wcB

with side dimensions 3m x 3m
