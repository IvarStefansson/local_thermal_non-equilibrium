import numpy as np
import porepy as pp


class DataSaving:
    def initialize_data_saving(self) -> None:
        super().initialize_data_saving()
        # Initialize data saving for temperature at monitoring points
        self.results = {
            "fluid_temperature": np.empty((0, 4)),
            "solid_temperature": np.empty((0, 3)),
        }

    def save_data_time_step(self) -> None:
        super().save_data_time_step()
        # Save temperature at monitoring points
        for sd in self.mdg.subdomains():
            inds = self.monitor_cells_fluid(sd)
            tempsf = self.fluid_temperature([sd]).value(self.equation_system)[inds]
            self.results["fluid_temperature"] = np.vstack(
                (self.results["fluid_temperature"], tempsf)
            )
            inds = self.monitor_cells_solid(sd)
            temps = self.solid_temperature([sd]).value(self.equation_system)[inds]
            self.results["solid_temperature"] = np.vstack(
                (self.results["solid_temperature"], temps)
            )

    def monitor_cells_fluid(self, sd: pp.GridLike) -> None:
        # Find indices of cells to monitor
        coords = (
            np.array(
                [
                    [0.125, 0.375, 0.625, 0.875],
                    [0.125, 0.375, 0.625, 0.875],
                    [0, 0, 0, 0],
                ]
            )
            * self.domain_size
        )
        cells = sd.closest_cell(coords)
        return cells

    def monitor_cells_solid(self, sd: pp.GridLike) -> None:
        # Find indices of cells to monitor
        coords = (
            np.array(
                [
                    [0.25, 0.5, 0.75],
                    [0.25, 0.5, 0.75],
                    [0, 0, 0],
                ]
            )
            * self.domain_size
        )
        cells = sd.closest_cell(coords)
        return cells


class DataSaving3d:
    def initialize_data_saving(self) -> None:
        super().initialize_data_saving()
        # Initialize data saving for temperature. Columns for time, x, and temperature.
        self.results = {
            "fluid_temperature": np.empty((0, 3)),
            "solid_temperature": np.empty((0, 3)),
        }

    def save_data_time_step(self) -> None:
        super().save_data_time_step()
        # Save temperature at monitoring points
        for sd in self.mdg.subdomains():
            self.save_average_temperatures(sd)

    def save_average_temperatures(self, sd: pp.GridLike) -> None:
        # Average temperatures for each slice between two x coordinates.
        # Get the temperature values
        tempsf = self.fluid_temperature([sd]).value(self.equation_system)
        tempss = self.solid_temperature([sd]).value(self.equation_system)
        # Get the coordinates
        x_cell = sd.cell_centers[0]
        # Get the unique x coordinates
        x_coords = np.linspace(0, self.domain_size, 4)
        # Compute the average temperatures
        for i in range(len(x_coords) - 1):
            inds = np.where((x_cell >= x_coords[i]) & (x_cell < x_coords[i + 1]))[0]
            avg_tempf = np.mean(tempsf[inds])
            avg_temps = np.mean(tempss[inds])
            time = self.time_manager.time
            x_mid = 0.5 * (x_coords[i] + x_coords[i + 1])
            self.results["fluid_temperature"] = np.vstack(
                (self.results["fluid_temperature"], [time, x_mid, avg_tempf])
            )
            self.results["solid_temperature"] = np.vstack(
                (self.results["solid_temperature"], [time, x_mid, avg_temps])
            )


class DataSavingLTE:
    def initialize_data_saving(self) -> None:
        super().initialize_data_saving()
        # Initialize data saving for temperature. Columns for time, x, and temperature.
        self.results = {"temperature": np.empty((0, 3))}

    def save_data_time_step(self) -> None:
        super().save_data_time_step()
        # Save temperature at monitoring points
        for sd in self.mdg.subdomains():
            self.save_average_temperature(sd)

    def save_average_temperature(self, sd: pp.GridLike) -> None:
        # Average temperatures for each slice between two x coordinates.
        # Get the temperature values
        temps = self.temperature([sd]).value(self.equation_system)
        # Get the coordinates
        x_cell = sd.cell_centers[0]
        # Get the unique x coordinates
        x_coords = np.linspace(0, self.domain_size, 4)
        # Compute the average temperatures
        for i in range(len(x_coords) - 1):
            inds = np.where((x_cell >= x_coords[i]) & (x_cell < x_coords[i + 1]))[0]
            avg_temp = np.mean(temps[inds])
            time = self.time_manager.time
            x_mid = 0.5 * (x_coords[i] + x_coords[i + 1])
            self.results["temperature"] = np.vstack(
                (self.results["temperature"], [time, x_mid, avg_temp])
            )
