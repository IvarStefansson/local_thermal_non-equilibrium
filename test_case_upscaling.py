import copy

import numpy as np
import porepy as pp

from common_params import (
    base_path,
    fluid_vals_3d,
    params_3d,
    solid_vals_3d,
    cells,
    dts,
)
from energy_balances import LTDFluid, LTDSolid
from model_setup import SingleDim3dModelLTNE

if __name__ == "__main__":
    # Set interface heat transfer coefficients corresponding to cases 3.1 and 3.2.
    interface_coefficients = [8.3e7, 100]
    for h in interface_coefficients:
        solid_vals_3d["interface_heat_transfer_coefficient"] = h
        solid = LTDSolid(**solid_vals_3d)
        fluid = LTDFluid(**fluid_vals_3d)

        def run_upscaling_model(nc: int, dt: float):
            """Run the model for a given number of cells and time step."""
            # Label the test case.
            case = f"upscaling_h_{h}_ncx_{nc}_dt_{dt}"
            params_loc = copy.deepcopy(params_3d)
            # Update the parameters with the current test case.
            params_loc.update(
                {
                    "material_constants": {"solid": solid, "fluid": fluid},
                    "num_cells_x": nc,
                    "time_manager": pp.TimeManager([0, 100], dt, constant_dt=True),
                    "times_to_export": [],
                }
            )
            # Run the model.
            model = SingleDim3dModelLTNE(params_loc)
            pp.run_time_dependent_model(model, params_loc)
            # Export the results.
            model.exporter.write_pvd()
            for phase, name in zip(["solid", "fluid"], ["solid", "void"]):
                pth = base_path + f"output_data_REV_{name}_{case}.csv"
                vals = model.results[phase + "_temperature"]
                np.savetxt(
                    pth, vals, delimiter=",", header="Time,X-coordinate,Temperature"
                )

        # Run the model for different number of cells and the smallest time step.
        for nc in cells:
            run_upscaling_model(nc, dts[-1])

        # Run the model for the smallest cells size and different time steps.
        for dt in dts:
            run_upscaling_model(cells[-1], dt)
