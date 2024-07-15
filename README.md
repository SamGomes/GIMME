# Welcome to the GIMME project repository

<img src="ReadmeImages/logo.png" width="300" alt="">

![version](https://img.shields.io/badge/version-2.0.0-red)
![version](https://img.shields.io/badge/python-v3.12-blue)

This repository contains the code for the ```GIMMECore``` Python package that is part of the GIMME (Group Interactions Management for Multiplayer sErious games) research project. 
GIMME focuses on the formation of work groups so that collective ability improves. What distinguishes this approach is that the interaction preferences of learners are explicitly considered when forming group configurations (also commonly named coalition structures).
The core of the application is included, along with some examples. 
Over time, we aim to improve the core functionalities as well as provide more examples for ```GIMMECore```.


Information about the API internals and examples can be consulted in our [wiki](https://github.com/SamGomes/GIMME/wiki).

## Requirements

```GIMMECore``` requires Python 3 in order to be executed (tested in Python 3.12.4). The package was tested on Windows and Linux. 


## Setup

The Python package is installed as usual:

```python 
pip install GIMMECore
```

*Note #1: The installed version may not correspond to the latest version, and so some aspects of the API may differ (especially relevant since the revision for version 2.0.0). It is advised to check our wiki in case of any naming doubt.*

*Note #2: If some errors about libraries are prompted (e.g., numpy or matplotlib package not found), please install those packages as well. We are currently reimplementing some code parts, and so we do not guarantee that the requirements are updated to the last code version.*

Once installed, programs can import the package with the following command:

```python 
from GIMMECore import *
```
This will automatically import all the associated classes.
Besides importing the core, the user has to implement functionalities to store and fetch data used by the library. This is done by extending two abstract data bridges: the [PlayerModelBridge](https://github.com/SamGomes/GIMME/wiki/PlayerModelBridge) and [TaskModelBridge](https://github.com/SamGomes/GIMME/wiki/TaskModelBridge). Consult our [wiki](https://github.com/SamGomes/GIMME/wiki) for more detail.

## Execute an example

Some examples are provided as use cases for our package. To execute the provided examples, you have to call Python as usual, for instance:

```python 
python examples/simpleExample/simpleExample.py
python examples/simulations/simulations.py
```

*Note: For just testing the code, it is advised to change the num_runs variable in simulations.py to a low value, such as 10. For tendencies to be clearly observed when executing them, it is advised to set num_runs to 200.*

The ```simulations.py``` example will output a result csv file ```/examples/simulations/analyzer/results/resultsXXXX.csv``` where XXXX symbolizes the id of the process that invoked the example. Several plots of the results can be built using the r code provided in ```/examples/simulations/analyzer/plotGenerator.r```.


## Future Improvements
As of the current version, there are still some on-going exploration pathways. They include:
- The addition and refinement of coalition structure generators (ConfigGenAlg);
- The improvement of task selection.

*Any help to improve this idea is welcome.*

## License
The current and previous versions of the code are licensed according to Attribution 4.0 International (CC BY 4.0).  
 
 <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />
