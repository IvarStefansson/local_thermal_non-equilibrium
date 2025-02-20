from typing import Literal

import numpy as np
import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)


class UnitSquare(SquareDomainOrthogonalFractures):
    def grid_type(self) -> Literal["simplex", "cartesian", "tensor_grid"]:
        """Grid type for the mixed-dimensional grid.

        Returns:
            Grid type for the mixed-dimensional grid.

        """
        return self.params.get("grid_type", "tensor_grid")

    def meshing_arguments(self) -> dict[str, float]:
        """Meshing arguments for mixed-dimensional grid generation.

        Returns:
            Meshing arguments compatible with
            :meth:`~porepy.grids.mdg_generation.create_mdg`.

        """
        # Default value of 1/2, scaled by the length unit.
        cell_size = 0.2 * self.domain_size
        default_meshing_args: dict[str, float] = {"cell_size": cell_size}
        num_cells_sides = 4  # Even number of points for odd number of cells
        num_cells_mid = 5 * 8  # Odd multiple of 8
        pts = np.hstack(
            (
                np.linspace(0, 1 / 16, num_cells_sides, endpoint=False),
                np.arange(1 / 16, 15 / 16, step=1 / num_cells_mid),
                np.linspace(15 / 16, 1, num_cells_sides),
            )
        )
        default_meshing_args["x_pts"] = pts * self.domain_size
        default_meshing_args["y_pts"] = pts * self.domain_size
        # If meshing arguments are provided in the params, they should already be
        # scaled by the length unit.
        user_args = self.params.get("meshing_arguments", {})
        # Update default meshing arguments with user arguments
        meshing_args = {**default_meshing_args, **user_args}
        return meshing_args


class RectangularCuboid:
    """A rectangular cuboid domain with no fractures.

    Dimensions: 45/28 cm x 15/28 cm x 15/28 cm.

    """

    def grid_type(self) -> Literal["simplex", "cartesian", "tensor_grid"]:
        """Grid type for the mixed-dimensional grid.

        Returns:
            Grid type for the mixed-dimensional grid.

        """
        return self.params.get("grid_type", "tensor_grid")

    @property
    def domain_size(self) -> float:
        """Return the size of the domain."""
        return 45 * self.unit_cell_size

    @property
    def unit_cell_size(self) -> float:
        """Return the size of a unit cell."""
        return self.units.convert_units(1 / 2800, "m")

    def set_domain(self) -> None:
        """Set the rectangular cuboid domain."""

        # Mono-dimensional grid by default
        yz = self.unit_cell_size * 4
        phys_dims = self.units.convert_units(np.array([self.domain_size, yz, yz]), "m")
        box = {"xmax": phys_dims[0], "ymax": phys_dims[1], "zmax": phys_dims[2]}
        self._domain = pp.Domain(box)  # 0 min values by default

    def meshing_arguments(self) -> dict[str, float]:
        """Meshing arguments for mixed-dimensional grid generation.

        Returns:
            Meshing arguments compatible with
            :meth:`~porepy.grids.mdg_generation.create_mdg` and a tensor grid.

        """
        n_c_yz = self.params.get("num_cells_yz", 1)
        n_c_x = self.params.get("num_cells_x", 45)
        r = self.domain_size / 45
        meshing_args = {
            "cell_size": r,
            "x_pts": np.linspace(0, self.domain_size, n_c_x + 1),
            "y_pts": np.linspace(0, 4 * self.unit_cell_size, n_c_yz + 1),
            "z_pts": np.linspace(0, 4 * self.unit_cell_size, n_c_yz + 1),
        }
        return meshing_args
