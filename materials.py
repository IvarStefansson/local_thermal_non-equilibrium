from dataclasses import dataclass
from typing import ClassVar

import porepy as pp


@dataclass(kw_only=True, eq=False)
class LTDFluid(pp.FluidComponent):
    SI_units: ClassVar[dict[str, str]] = pp.FluidComponent.SI_units
    SI_units.update({"specific_pore_contact_area": "-"})
    specific_pore_contact_area: 1.0


@dataclass(kw_only=True, eq=False)
class LTDSolid(pp.SolidConstants):
    """Solid material constants for local thermal dynamics."""

    SI_units: ClassVar[dict[str, str]] = pp.SolidConstants.SI_units
    SI_units.update(
        {
            "energy_exchange_number": "J*kg^-1*K^-1",
            "interface_heat_transfer_coefficient": "W*m^-2*K^-1",
            "pore_size": "m",
            "fluid_solid_interfacial_area": "m^-1",
            "specific_grain_contact_area": "m^2*m^-2",
            "nuske_f_e": "-",
        }
    )
    energy_exchange_number: pp.number = 1.0
    pore_size: pp.number = 1.0
    fluid_solid_interfacial_area: pp.number = 1.0
    specific_grain_contact_area: pp.number = 1.0
    interface_heat_transfer_coefficient: pp.number = 1.0
    nuske_f_e: pp.number = 0.5
