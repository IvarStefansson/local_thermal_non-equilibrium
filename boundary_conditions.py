import numpy as np
import porepy as pp


class BoundaryConditionsMassAndEnergyDirWestEast(
    pp.mass_and_energy_balance.BoundaryConditionsFluidMassAndEnergy
):
    r"""Boundary conditions for the thermoporomechanics problem.

    Dirichlet boundary conditions are defined on the north and south boundaries. Some
    of the default values may be changed directly through attributes of the class.

    Implementation of mechanical values facilitates time-dependent boundary conditions
    with use of :class:`pp.time.TimeDependentArray` for :math:`\nabla \cdot u` term.

    Usage: tests for models defining equations for any subset of the thermoporomechanics
    problem.

    """

    def in_faces(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Faces with inflow boundary conditions."""
        if bg.dim == self.nd - 2:
            return self.domain_boundary_sides(bg).west
        else:
            return np.array([], dtype=int)

    def out_faces(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Faces with inflow boundary conditions."""
        if bg.dim == self.nd - 1:
            return self.domain_boundary_sides(bg).east
        else:
            return np.array([], dtype=int)

    @property
    def inflow_temperature(self):
        return self.reference_variable_values.temperature + self.units.convert_units(
            10, "K"
        )

    @property
    def inflow_pressure(self):
        val = self.reference_variable_values.pressure
        return val

    def dir_faces(self, sd):
        """Faces with Dirichlet boundary conditions."""
        if sd.dim == self.nd - 1:
            return (
                self.domain_boundary_sides(sd).west
                + self.domain_boundary_sides(sd).east
            )
        else:
            return np.array([], dtype=int)

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for Darcy flux.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, self.dir_faces(sd), "dir")

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Boundary condition values for Darcy flux.

        Dirichlet boundary conditions are defined on the north and south boundaries,
        with a constant value of 0 unless fluid's reference pressure is changed.

        Parameters:
            subdomains: List of subdomains for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        vals = np.ones(bg.num_cells) * self.reference_variable_values.pressure
        vals[self.in_faces(bg)] = self.inflow_pressure
        return vals

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary conditions on all external boundaries.

        Parameters:
            sd: Subdomain grid on which to define boundary conditions.

        Returns:
            Boundary condition object. Per default Dirichlet-type BC are assigned.

        """
        # Define boundary condition on all boundary faces.
        return pp.BoundaryCondition(sd, self.dir_faces(sd), "dir")

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for the Fourier heat flux.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, self.dir_faces(sd), "dir")

    def bc_type_solid_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for the Fourier heat flux.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, self.dir_faces(sd), "dir")

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for the enthalpy.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, self.dir_faces(sd), "dir")

    def bc_values_temperature(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Boundary condition values for the FLUID Fourier heat flux."""
        # Get density and viscosity values on boundary faces applying trace to
        # interior values.
        # Append to list of boundary values
        vals = np.ones(bg.num_cells) * self.reference_variable_values.temperature
        vals[self.in_faces(bg)] = self.inflow_temperature

        return vals

    def bc_values_solid_temperature(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Boundary condition values for the Fourier heat flux."""
        # Get density and viscosity values on boundary faces applying trace to
        # interior values.
        # Append to list of boundary values
        vals = np.ones(bg.num_cells) * self.reference_variable_values.temperature
        return vals


class SingleDimBCs(BoundaryConditionsMassAndEnergyDirWestEast):
    def in_faces(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Faces with inflow boundary conditions."""
        return self.domain_boundary_sides(bg).west

    def out_faces(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Faces with outflow boundary conditions."""
        return self.domain_boundary_sides(bg).east

    def dir_faces(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Faces with Dirichlet boundary conditions."""
        return self.in_faces(bg) + self.out_faces(bg)


class NeumannSolidFourier:
    def bc_type_solid_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for the Fourier heat flux.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, cond="neu")


class BCs3d(NeumannSolidFourier, SingleDimBCs):
    def out_faces(self, grid: pp.Grid | pp.BoundaryGrid) -> np.ndarray:
        """Faces with outflow boundary conditions."""
        if isinstance(grid, pp.BoundaryGrid):
            return np.zeros(grid.parent.num_faces, dtype=bool)
        else:
            return np.zeros(grid.num_faces, dtype=bool)
