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
from energy_balances import (
    LTDFluid,
    LTDSolid,
    ThermalParametersNuske,
    ThermalParametersNakayama,
    AdditonalTermNakayama,
)
from model_setup import SingleDim3dModelLTNE


class Nuske(ThermalParametersNuske, SingleDim3dModelLTNE):
    pass


class Nakayama(ThermalParametersNakayama, AdditonalTermNakayama, SingleDim3dModelLTNE):
    pass


model_dict = {"Nakayama": Nakayama, "Nuske": Nuske}
if __name__ == "__main__":
    solid = LTDSolid(**solid_vals_3d)
    fluid = LTDFluid(**fluid_vals_3d)
    for model_name, model_class in model_dict.items():

        def run_model(nc, dt):
            case = f"{model_name}_ncx_{nc}_dt_{dt}"
            params_loc = copy.deepcopy(params_3d)
            params_loc.update(
                {
                    "material_constants": {"solid": solid, "fluid": fluid},
                    "folder_name": f"results/{case}",
                    "num_cells_x": nc,
                    "time_manager": pp.TimeManager([0, 100], dt, constant_dt=True),
                    "times_to_export": [],
                }
            )
            model = model_class(params_loc)

            pp.run_time_dependent_model(model, params_loc)
            model.exporter.write_pvd()
            for phase, name in zip(["solid", "fluid"], ["solid", "void"]):
                pth = base_path + f"output_data_REV_{name}_{case}.csv"
                vals = model.results[phase + "_temperature"]
                np.savetxt(
                    pth, vals, delimiter=",", header="Time,X-coordinate,Temperature"
                )

        for nc in cells:
            run_model(nc, dts[-1])

        for dt in dts:
            run_model(cells[-1], dt)
