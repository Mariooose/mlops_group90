# pokemon_classification

The goal of this project is to make a machine learning model that takes in an image of a pokemon and classifies the pokemon.

# Framework

For this project we intend to use the pytorch image model (timm)

# Descrioption of data

The data consists of 26000+ images of 1000 different pokémon. Each image is a 128x128 pixel PNG file, some are in black and white and some are in color. The data can be found at the [link](https://www.kaggle.com/datasets/noodulz/pokemon-dataset-1000/data). 

The data is divided into different folders of training, testing and validation images of the different pokemons. Each of theese folders then have subfolders each with images of a said pokemon. 

# Model

We intend to use a neural network based on the ResNet network from the framwork timm, as this model has shown to be usefull in other projects with data like ours. This is our current idea, but as we are not very familiar with theese kind of models our aproach may change later.

# Members of the group
The members in the group are:  
-s224388  
-s224360  
-s224401  


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
