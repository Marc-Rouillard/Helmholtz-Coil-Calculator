# Square Helmholtz Coil Calculator

This is a basic calculator for analysing field strength and uniformity in a square Helmholtz coil, using matplotlib.

![alt text](images/inputs.png)

![alt text](images/plots.png)

![alt text](images/printouts.png)

## Usage
Clone the repo from GitHub and open the folder in a terminal. Activate a virtual environment if you wish and then run:

`pip install -r requirements.txt`

Once you have installed the necessary packages you can edit the inputs in `calculate.py` and run the program.

`python src/calculate.py`

The matplotlib layout manager is not perfect so you may need increase the window size to avoid overlapping plots.

## Limitations and Future Improvement
* The current approach to calculating the field strength and obtaining region of interest parameters is fairly basic and could certainly be optimised to improve calculation times.
* The current implementation is limited to square coils. Alternative implementations could extend the simulation to circular and irregularly shaped coils, although these may require substantial changes to the approach to be feasible
