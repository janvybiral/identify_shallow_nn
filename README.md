# identify_shallow_nn


## Installation

The core modules use python 3. It is best to install the module in its own virtual environment.

Here, we use [conda](https://docs.conda.io/en/latest/), but most other environment managers should work as well. 

First we create a new environment called *snnident*
```
$ conda create -n snnident python=3.8
$ conda activate snnident
```
Now, clone the repository 
```
(snnident)$ git clone https://github.com/michaelrst/identify_shallow_nn
```
this should download the repository into the current working directory, which can now be installed together with the dependencies by
```
(snnident)$ pip install -e identify_shallow_nn
```



## Running the example notebooks

To run examples we need to install jupyter notebook or jupyter lab. 
```
conda install jupyterlab
```

Lastly, we have to add the enviroment to jupyterlab

```
(snnident)$ conda install ipykernel
(snnident)$ ipython kernel install --user --name=snnident
```

To run the notebooks, simply run the following command and navigate to notebooks/

```
(snnident)$ jupyter lab 
```
