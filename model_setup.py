import porepy as pp

from boundary_conditions import BCs3d, SingleDimBCs
from energy_balances import (
    FluidMassAndEnergyBalanceLTNE,
    NonzeroInitialCondition,
)
from geometry import RectangularCuboid
from visualization import DataSaving, DataSaving3d, DataSavingLTE


class SingleDimModel(  # type: ignore
    SingleDimBCs,
    pp.constitutive_laws.CubicLawPermeability,
    DataSaving,
    FluidMassAndEnergyBalanceLTNE,
):
    pass


class SingleDim3dModelLTNE(  # type: ignore
    RectangularCuboid,
    BCs3d,
    DataSaving3d,
    FluidMassAndEnergyBalanceLTNE,
):
    pass


class LTEAdjustments(NonzeroInitialCondition):
    def thermal_conductivity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Thermal conductivity [m^2].

        The thermal conductivity is computed as the porosity-weighted average of the
        fluid and solid thermal conductivities. In this implementation, both are
        considered constants, however, if the porosity changes with time, the weighting
        factor will also change.

        Parameters:
            subdomains: List of subdomains where the thermal conductivity is defined.

        Returns:
            Cell-wise conductivity operator.

        """
        nc = sum(sd.num_cells for sd in subdomains)
        conductivity = pp.wrap_as_dense_ad_array(
            1, nc
        ) * self.solid_thermal_conductivity(subdomains)
        return self.isotropic_second_order_tensor(subdomains, conductivity)

    def fluid_internal_energy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Internal energy of the fluid.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the fluid energy.

        """
        energy = (
            self.fluid.density(subdomains) * self.fluid.specific_enthalpy(subdomains)
        ) * self.porosity(subdomains)
        energy.set_name("fluid_internal_energy")
        return energy


class SingleDim3dModelLTE(  # type: ignore
    RectangularCuboid,
    BCs3d,
    DataSavingLTE,
    LTEAdjustments,
    pp.models.mass_and_energy_balance.MassAndEnergyBalance,
):
    """Model for the LTE test case."""
