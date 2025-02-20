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
from model_setup import SingleDim3dModelLTE

if __name__ == "__main__":
    # The solid value is used directly in the model for the effective conductivity, see
    # LTEAdjustments.
    solid_vals_3d["thermal_conductivity"] = 2.411111344

    def run_lte_model(nc, dt):
        solid = LTDSolid(**solid_vals_3d)
        fluid = LTDFluid(**fluid_vals_3d)
        params_loc = copy.deepcopy(params_3d)
        case = f"ncx_{nc}_dt_{dt}"

        params_loc.update(
            {
                "material_constants": {"solid": solid, "fluid": fluid},
                "folder_name": f"results/test_case_lte_{case}",
                "file_name": case,
                "num_cells_x": nc,
                "num_cells_yz": 1,
                "time_manager": pp.TimeManager([0, 100], dt, constant_dt=True),
                "times_to_export": np.linspace(0, 100, 6),
            }
        )

        model = SingleDim3dModelLTE(params_loc)
        pp.run_time_dependent_model(model, params_loc)
        model.exporter.write_pvd()
        pth = base_path + f"output_data_REV_LTE_{case}.csv"
        vals = model.results["temperature"]
        np.savetxt(pth, vals, delimiter=",", header="Time,X-coordinate,Temperature")

    for nc in cells:
        run_lte_model(nc, dts[-1])

    for dt in dts:
        run_lte_model(cells[-1], dt)
