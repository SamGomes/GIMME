# Welcome to the GIMME project repository

<img src="ReadmeImages/logo.png" width="300">

![version](https://img.shields.io/badge/version-1.1.5-blue)
![version](https://img.shields.io/badge/python-v3.7-blue)

GIMME (Group Interactions Management for Multiplayer sErious games) is a research tool which focuses on the management of 
interactions in groups so that the collective ability improves. 
More specifically, what sets the method appart is that the interactions between players are explicity considered when forming group configurations (also commonly named coalition structures).
This repository contains the core of the application (written in Python), as well as some examples. 
Over time, we aim to improve the core functionalities as well as provide more examples for GIMME.


Information about the API internals and examples can be observed in our [wiki](https://github.com/SamGomes/GIMME/wiki).

## Requirements

GIMME requires Python 3 in order to be executed (Tested in Python 3.7.0). No additional tools are required.
GIMME was tested on Windows and Linux. May also work in MacOS, but remains untested in this platform.


## Setup

GIMME setup is straightforward. You just got to install the python package:

```python 
sudo python3 setup.py install
```

*Note: If some errors about libraries are prompted (for ex. numpy or matplotlib package not found), please install those packages as well, we are currently reimplementing some code parts, by which we do not ensure the requirements are updated to the last code version...*

Then you can start to write programs with our library.
When importing the package, it is recommended to use the following command:

```python 
from GIMMECore import *
```
This will automatically import all of the associated GIMME classes.
Besides importing the core, the user has to also implement the functionalities to store data used by the algorithm. This is done by extending two abstract data bridges: the [PlayerModelBridge](https://github.com/SamGomes/GIMME/wiki/PlayerModelBridge) and [TaskModelBridge](https://github.com/SamGomes/GIMME/wiki/TaskModelBridge). 

## Execute an example

Some simulations are provided as example use cases of our system. To execute the provided simulations, you just have to enter the folder examples and call python as usual:

```python 
cd examples
python simulations.py
```

*Note: For just testing the code, it is advised to change the numRuns variable to a low value, higher than 1, such as 10. For tendencies to be clearly observed when executing them, it is adviseable to set numRuns to 200.*

This will output the data to a csv file ```examples/simulationResults/latestResults/GIMMESims/results.csv)```, summing the results of applying our simulations. Several plots summarizing the results can be built using the r code provided in ```examples/simulationResults/latestResults/plotGenerator.r```.

## Unit Tests
As of now, no unit tests are provided, due to development constraints. We believe unit tests will be provided in near updates, as they are needed to ensure consistency in future releases.


## Report on Latest Features
We have been writing a report about our latest features, currently available [here](https://drive.google.com/file/d/1Qa5Mccx_P4rPGLnM6NnZ9IXBemLdd3Zp/view?usp=sharing) in pre-print format.

## Future Improvements
As of the current version, there are still some unexplored open pathways. They include:
- The integration of more refined coalition structure generators (ConfigGenAlg);
- The integration of the tool in a multiplayer serious game (example/ use case);
- The integration of automatic task selection (not covered by simulations).

*Any help to improve this idea is welcome.*


## License
The current and previous versions of the code are licensed according to Attribution 4.0 International (CC BY 4.0).  
 
 <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />
