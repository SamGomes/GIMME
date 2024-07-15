# Welcome to the GIMME project repository

<img src="ReadmeImages/logo.png" width="300" alt="">

![version](https://img.shields.io/badge/version-2.0.0-blue)
![version](https://img.shields.io/badge/python-v3.11-blue)

GIMME (Group Interactions Management for Multiplayer sErious games) is a research framework that focuses on the formation of work groups so that collective ability improves. 
What distinguishes this approach is that the interaction preferences of learners are explicitly considered when forming group configurations (also commonly named coalition structures).
This repository contains the core of the application (written in Python) as well as some examples. 
Over time, we aim to improve the core functionalities as well as provide more examples for the ```GIMMECore``` API.


Information about the API internals and examples can be observed in our [wiki](https://github.com/SamGomes/GIMME/wiki).

## Requirements

GIMME requires Python 3 in order to be executed (tested in Python 3.11.6). 
GIMME was tested on Windows and Linux. 


## Setup

The setup is straightforward. You just have to install the Python package via the repository:

```python 
pip install GIMMECore
```

*Note #1: The installed version may not correspond to the latest version, and so some aspects of the API may differ (especially relevant since the revision for version 2.0.0). It is advised to check our wiki in case of any naming doubt.*

*Note #2: If some errors about libraries are prompted (for ex., numpy or matplotlib package not found), please install those packages as well. We are currently reimplementing some code parts, and so we do not guarantee that the requirements are updated to the last code version...*

Then you can start to write programs with our library.
When importing the package, it is recommended to use the following command:

```python 
from GIMMECore import *
```
This will automatically import all of the associated GIMME classes.
Besides importing the core, the user has to also implement the functionalities to store data used by the algorithm. This is done by extending two abstract data bridges: the [PlayerModelBridge](https://github.com/SamGomes/GIMME/wiki/PlayerModelBridge) and [TaskModelBridge](https://github.com/SamGomes/GIMME/wiki/TaskModelBridge). 

## Execute an example

Some examples are provided as use cases for our package. To execute the provided examples, you just have to call Python as usual, for instance:

```python 
python examples/simpleExample/simpleExample.py
python examples/simulations/simulations.py
```

*Note: For just testing the code, it is advised to change the numRuns variable in simulations.py to a low value, such as 10. For tendencies to be clearly observed when executing them, it is adviseable to set numRuns to 200.*

The ```simulations.py``` example will output the data to a csv file ```examples/simulationResults/latestResults/GIMMESims/results.csv```, summarizing the results of applying our simulations. Several plots of the results can be built using the r code provided in ```examples/simulationResults/latestResults/plotGenerator.r```.


## Future Improvements
As of the current version, there are still some on-going exploration pathways. They include:
- The integration of more refined coalition structure generators (ConfigGenAlg);
- The improvement of task selection.

*Any help to improve this idea is welcome.*

## License
The current and previous versions of the code are licensed according to Attribution 4.0 International (CC BY 4.0).  
 
 <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />
