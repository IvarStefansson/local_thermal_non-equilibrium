import porepy as pp

T_0 = 273.15 + 20
solid_vals = {
    "permeability": 7.321e-11,
    "thermal_conductivity": 2.8,
    "density": 2.7e3,
    "energy_exchange_number": 0.5,
    "porosity": 0.2137,
    "pore_size": 3.5734e-4,
    "specific_grain_contact_area": 1,
    "specific_heat_capacity": 7.9e2,
}
solid_vals_3d = solid_vals.copy()
solid_vals_3d.update(
    {
        "permeability": 4.13e-11,
        "energy_exchange_number": 0.5,
        "porosity": 0.25446,
        "pore_size": 3.58586e-4,
        "fluid_solid_interfacial_area": 6511.1503,
        "specific_grain_contact_area": 0.863663,
    }
)

fluid_vals = {
    "compressibility": 0,
    "density": 1e3,
    "thermal_conductivity": 0.679,
    "normal_thermal_conductivity": 0.679,
    "specific_heat_capacity": 4.18e3,
    "viscosity": 1e-3,
}
fluid_vals_3d = fluid_vals.copy()
fluid_vals_3d["specific_pore_contact_area"] = 0.1087
reference_variable_values = pp.ReferenceVariableValues(pressure=1e5, temperature=T_0)
domain_size = 1e-2
params = {
    "domain_size": domain_size,
    "file_name": "ltd",
    "max_iterations": 50,
    "nl_convergence_tol": 1e-8,
    "nl_convergence_tol_res": 1e-10,
    "export_constants_separately": False,
    "meshing_arguments": {"cell_size": domain_size / 10},
    "fracture_indices": [],
    "reference_variable_values": reference_variable_values,
}
params_3d = {
    "file_name": "ltd_3d",
    "max_iterations": 10,
    "nl_convergence_tol": 1e-10,
    "nl_convergence_tol_res": 1e-6,
    "export_constants_separately": False,
    "fracture_indices": [],
    "time_manager": pp.TimeManager([0, 100], 10, constant_dt=True),
    "reference_variable_values": reference_variable_values,
}
base_path = "results/"
cells = [45 * 2**i for i in range(6)]
dts = [10 * 2.0 ** (-i) for i in range(6)]
