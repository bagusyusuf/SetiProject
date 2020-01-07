# SetiProject

### prerequisite

first of all u need to install all dependecies

- python 3.X
- numpy
- pandas
- sklearn
- tensorflow 2.X
- PIL

to run the simulation, you will need

- tkinter
- cv2

### training

run

```
python3 train.py
```

this training process will generate "mymodel.h5" model

or you can run jupyternotebook and open notebookVersion.ipynb
this file will generate model called "my_model_89.h5"

### testing

open notebookVersion_test.ipynb with jupyternotebook
and run all

by default this task will test "my_model_89.h5" model from training with jupyternotebook,
if you train the model with the first option and want to test the model
please change the model name on test section

### simulation

run this

```
python3 simulation.py
```
