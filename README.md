# Local Thermal Non-Equilibrium

This repository contains code for the REV scale simulations for the manuscript "Comparison study for Local Thermal Non-Equilibrium porous medium model concepts from the pore to REV-scale - Investigation of conduction".

The code relies on the PorePy library, which can be found at [PorePy GitHub Repository](https://github.com/pmgbergen/porepy/). The results were run on the develop branch of PorePy, commit hash `628c26c96f07e01481741155b0f93621b1498ddc`.

The code is structured as follows:
- `test_case_lte.py`: Contains the test case for the REV scale simulations using the Local Thermal Equilibrium model.
- `test_case_upscaling.py`: Contains the test case for the REV scale simulations using the Local Thermal Non-Equilibrium model with upscaled thermal properties.
- `test_case_nuske_nakayama.py`: Contains the test case for the REV scale simulations using the Local Thermal Non-Equilibrium model with thermal properties computed according to Nuske et al and Nakayama et al. as explained in the manuscript.
- `energy_balances.py`: Contains model extensions relative to the PorePy library to compute the energy balances for the different models, including the Local Thermal Non-Equilibrium model.
- The remaining files contain boundary conditions, geometry, parameters for the simulations etc.

Running the three first files (python test_case_*.py) will generate the results as reported in the manuscript for the different spatial and temporal refinement levels. The results will be saved in the `results` folder in files containing temperatures at different spatial locations at each time step. Post-processing of the results can be done using scripts in other repositories referenced in the manuscript.