# ML-GroupProject1

Repo for the first group project in the Fall 2017 EPFL Machine Learning course.

## Data
To run the project the data must be in the folder `data` with the names `train` and `test` in `.csv` format.

## Exploratory Analysis
The exploratory analysis are in the folder `data-analysis`.

## Install `mlcomp` 
There are 2 main way to install and run the application `mlcomp`

1. Install the project and also the dependencies in your machine, you must run:
```bash
pip3 install -e .
```

in the directory where is the file `setup.py`.

2. Create a virtual environment - it is a way to have separeted place to specify the dependencies to not mess with the dependencies installed in your machine - to run the application, you must run:
```bash
 bash bootstrap-python-env.sh
```
and then to active the environment, you must run:
```bash
source mlcomp-python-env/bin/activate
```

To check if the package is installed in your machine try: `pip3 freeze | grep mlcomp`, if it return `0.0.1` you are good to go =)


## Running the application
```bash
python3 mlcomp/src/mlcomp/main.py
```