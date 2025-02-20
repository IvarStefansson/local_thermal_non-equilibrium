"""Energy balance with advection and diffusion.

In the nd matrix, local thermal dynamics is assumed, i.e., the solid and fluid
temperatures are not assumed to be equal. However, the fractures are assumed to
be open, resulting in a single energy balance equation, for the fluid, for d<nd.

"""

from __future__ import annotations

from typing import Callable, Literal, Optional

import numpy as np
import porepy as pp
import scipy.sparse as sps
from porepy.applications.discretizations.flux_discretization import FluxDiscretization
from porepy.models.protocol import PorePyModel
from materials import LTDFluid, LTDSolid


class EnergyBalanceEquationsLTNE(pp.energy_balance.TotalEnergyBalanceEquations):
    """Mixed-dimensional energy balance equation.

    Balance equation for all subdomains and advective and diffusive fluxes internally
    and on all interfaces of codimension one.

    The class is not meant to be used stand-alone, but as a mixin in a coupled model.

    """

    # Expected attributes for this mixin.
    solid_enthalpy: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Solid enthalpy. Defined in a mixin class with a suitable constitutive relation.
    """
    solid_density: Callable[[list[pp.Grid]], pp.ad.Scalar]
    """Solid density. Defined in a mixin class with a suitable constitutive relation.
    """
    solid_specific_heat_capacity: Callable[[list[pp.Grid]], pp.ad.Scalar]
    """Solid specific heat capacity. Defined in a mixin class with a suitable constitutive
    relation.
    """
    solid_thermal_conductivity: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Solid thermal conductivity. Defined in a mixin class with a suitable constitutive
    relation.
    """
    porosity: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Porosity of the rock. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.ConstantPorosity` or a subclass thereof.

    """

    def set_equations(self):
        """Set the equations for the energy balance problem.

        A energy balance equation is set for each subdomain, and advective and diffusive
        fluxes are set for each interface of codimension one.

        """
        subdomains = self.mdg.subdomains()
        nd_subdomains = self.mdg.subdomains(dim=self.nd)
        interfaces = self.mdg.interfaces(codim=1)
        codim_2_interfaces = self.mdg.interfaces(codim=2)

        sd_eq_f = self.fluid_energy_balance_equation(subdomains)
        sd_eq_s = self.solid_energy_balance_equation(nd_subdomains)
        # Set the equations for the interface fluxes for the fluid phase.
        intf_cond = self.interface_fourier_flux_equation(interfaces)
        intf_adv = self.interface_enthalpy_flux_equation(interfaces)
        well_eq = self.well_enthalpy_flux_equation(codim_2_interfaces)

        self.equation_system.set_equation(sd_eq_f, subdomains, {"cells": 1})
        self.equation_system.set_equation(sd_eq_s, nd_subdomains, {"cells": 1})
        self.equation_system.set_equation(intf_cond, interfaces, {"cells": 1})
        self.equation_system.set_equation(intf_adv, interfaces, {"cells": 1})
        self.equation_system.set_equation(well_eq, codim_2_interfaces, {"cells": 1})

    def fluid_energy_balance_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Fluid energy balance equation for subdomains.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the energy balance equation.

        """
        accumulation = self.volume_integral(
            self.fluid_internal_energy(subdomains), subdomains, dim=1
        )
        flux = self.fourier_flux(subdomains) + self.enthalpy_flux(subdomains)
        # fluid-solid only in nd
        projection = pp.ad.SubdomainProjections(subdomains, dim=1)
        nd_subdomains = [sd for sd in subdomains if sd.dim == self.nd]
        f_s = projection.cell_prolongation(
            nd_subdomains
        ) @ self.fluid_solid_energy_exchange(nd_subdomains)
        source = self.energy_source(subdomains) + f_s
        eq = self.balance_equation(subdomains, accumulation, flux, source, dim=1)
        eq.set_name("fluid_energy_balance_equation")
        return eq

    def solid_energy_balance_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Solid energy balance equation for nd subdomains.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the energy balance equation.

        """
        accumulation = self.volume_integral(
            self.solid_internal_energy(subdomains), subdomains, dim=1
        )
        flux = self.solid_fourier_flux(subdomains)
        source = pp.ad.Scalar(-1) * self.fluid_solid_energy_exchange(subdomains)
        eq = self.balance_equation(subdomains, accumulation, flux, source, dim=1)
        eq.set_name("solid_energy_balance_equation")
        return eq

    def fluid_solid_energy_exchange(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Energy exchanged from fluid to solid phase.


        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the exchange term.

        """
        op = self.gradient_energy_exchange(subdomains)
        return self.volume_integral(op, subdomains, dim=1)

    def fluid_internal_energy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Internal energy of the fluid.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the fluid energy.

        """
        energy = (
            self.fluid.density(subdomains) * self.fluid.specific_enthalpy(subdomains)
            # - self.pressure(subdomains)
        ) * self.porosity(subdomains)
        energy.set_name("fluid_internal_energy")
        return energy


class FouriersLawLTNE(pp.constitutive_laws.FouriersLaw):
    """This class could be refactored to reuse for other diffusive fluxes. It's somewhat
    cumbersome, though, since potential, discretization, and boundary conditions all
    need to be passed around. Also, gravity effects are not included, as opposed to the
    Darcy flux (see that class).
    """

    bc_data_solid_fourier_flux_key: str = "solid_fourier_flux"
    """Keyword for the storage of Neumann-type boundary conditions for the Fourier
    flux."""

    solid_temperature: Callable[[list[pp.GridLike]], pp.ad.Variable]

    bc_type_solid_fourier_flux: Callable[[pp.Grid], pp.BoundaryCondition]

    solid_fourier_keyword: str

    reference_porosity: Callable[[list[pp.Grid]], pp.ad.Operator]

    porosity: Callable[[list[pp.Grid]], pp.ad.Operator]

    def solid_fourier_flux(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Discrete Fourier flux on subdomains.

        .. note::
            The below implementation assumes the heat flux is discretized with a finite
            volume method (either Tpfa or Mpfa). Other discretizations may be possible,
            but would likely require a modification of this (and several other) methods.

        Parameters:
            subdomains: List of subdomains where the Fourier flux is defined.

        Returns:
            An Ad-operator representing the Fourier flux on the subdomains.

        """
        if len(subdomains) == 0 or isinstance(subdomains[0], pp.BoundaryGrid):
            # Given Neumann data prescribed for Fourier flux on boundary.
            return self.create_boundary_operator(  # type: ignore[call-arg]
                name=self.bc_data_solid_fourier_flux_key, domains=subdomains
            )

        discr = self.solid_fourier_flux_discretization(subdomains)

        # As opposed to darcy_flux in :class:`DarcyFluxFV`, the gravity term is not
        # included here.
        boundary_operator_fourier = self._combine_boundary_operators(
            subdomains=subdomains,
            dirichlet_operator=self.solid_temperature,
            neumann_operator=self.solid_fourier_flux,
            robin_operator=None,
            bc_type=self.bc_type_solid_fourier_flux,
            name="bc_values_" + self.bc_data_solid_fourier_flux_key,
        )
        flux: pp.ad.Operator = (
            discr.flux() @ self.solid_temperature(subdomains)
            + discr.bound_flux() @ boundary_operator_fourier
        )
        flux.set_name("solid_Fourier_flux")
        return flux

    def solid_fourier_flux_discretization(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.MpfaAd:
        """Fourier flux discretization.

        Parameters:
            subdomains: List of subdomains where the Fourier flux is defined.

        Returns:
            Discretization object for the Fourier flux.

        """
        return pp.ad.MpfaAd(self.solid_fourier_keyword, subdomains)

    def _porosity_as_array(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        phi = self.porosity(subdomains)
        try:
            phi.value(self.equation_system)
        except KeyError:
            # We assume this means that the porosity includes a discretization matrix
            # for div_u which has not yet been computed.
            phi = self.reference_porosity(subdomains)
        if isinstance(phi, pp.ad.Scalar):
            size = sum([sd.num_cells for sd in subdomains])
            phi = phi * pp.wrap_as_dense_ad_array(1, size)
        return phi

    def gradient_energy_exchange(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Energy exchanged from fluid to solid phase.


        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the exchange term.

        """
        lbda_eff = self.energy_exchange_coefficient(subdomains)
        op = lbda_eff * (
            self.solid_temperature(subdomains) - self.fluid_temperature(subdomains)
        )
        op.set_name("fluid_solid_energy_exchange")
        return op


class ThermalParametersNakayama(PorePyModel):
    """Thermal conductivity in the local thermal dynamics.

    Interacts with the rest of the model through calls in set_discretization_parameters.
    """

    porosity: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Porosity of the rock. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.ConstantPorosity` or a subclass thereof.

    """
    reference_porosity: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Reference porosity of the rock. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.PoroMechanicsPorosity`.

    """
    unit_fourier_keyword_solid: str = "unit_fourier_keyword_solid"

    unit_fourier_keyword_fluid: str = "unit_fourier_keyword_fluid"

    def _turtuoisity_G(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        k_f = pp.ad.Scalar(self.fluid.reference_component.thermal_conductivity)
        sigma = self._sigma(subdomains)
        k_stg = self._stagnant_thermal_conductivity(subdomains)
        phi = self._phi(subdomains)
        one = pp.ad.Scalar(1)
        G = ((k_stg / k_f) - phi - (one - phi) * sigma) / (sigma - one) ** 2
        G.set_name("turtuosity_G")
        return G

    def _sigma(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        k_s = self.solid.thermal_conductivity
        k_f = self.fluid.reference_component.thermal_conductivity
        return pp.ad.Scalar(k_s / k_f)

    def _phi(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        phi = self.porosity(subdomains)
        try:
            phi.value(self.equation_system)
        except KeyError:
            # We assume this means that the porosity includes a discretization matrix
            # for div_u which has not yet been computed.
            phi = self.reference_porosity(subdomains)
        if isinstance(phi, pp.ad.Scalar):
            size = sum([sd.num_cells for sd in subdomains])
            phi = phi * pp.wrap_as_dense_ad_array(1, size)
        return phi

    def _stagnant_thermal_conductivity(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        sigma = self._sigma(subdomains)
        phi = self._phi(subdomains)
        one = pp.ad.Scalar(1)
        one_minus_phi = one - phi
        k_stg = pp.ad.Scalar(self.fluid.reference_component.thermal_conductivity) * (
            one
            - one_minus_phi ** pp.ad.Scalar(2 / 3)
            + one_minus_phi ** pp.ad.Scalar(2 / 3)
            * sigma
            / (
                (one - one_minus_phi ** pp.ad.Scalar(1 / 3)) * sigma
                + one_minus_phi ** pp.ad.Scalar(1 / 3)
            )
        )

        k_stg.set_name("stagnant_thermal_conductivity")
        return k_stg

    def fluid_thermal_conductivity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Thermal conductivity [m^2].

        Parameters:
            subdomains: List of subdomains where the thermal conductivity is defined.

        Returns:
            Cell-wise conductivity operator.

        """
        phi = self._phi(subdomains)
        G = self._turtuoisity_G(subdomains)
        sigma = self._sigma(subdomains)
        k_f = pp.ad.Scalar(self.fluid.reference_component.thermal_conductivity)
        conductivity = (phi + (1 - sigma) * G) * k_f
        conductivity.set_name("fluid_thermal_conductivity")
        return self.isotropic_second_order_tensor(subdomains, conductivity)

    def solid_thermal_conductivity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Thermal conductivity [m^2].

        Parameters:
            subdomains: List of subdomains where the thermal conductivity is defined.

        Returns:
            Cell-wise conductivity operator.

        """
        phi = self._phi(subdomains)
        G = self._turtuoisity_G(subdomains)
        sigma = self._sigma(subdomains)
        k_s = pp.ad.Scalar(self.solid.thermal_conductivity)
        one = pp.ad.Scalar(1)
        conductivity = (one - phi + (sigma - one) * G) * k_s
        conductivity.set_name("solid_thermal_conductivity")
        return self.isotropic_second_order_tensor(subdomains, conductivity)

    def energy_exchange_coefficient(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.DenseArray:
        """Energy exchange coefficient.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the energy exchange coefficient.

        """
        a = pp.ad.Scalar(self.solid.fluid_solid_interfacial_area)
        a.set_name("fluid_solid_interfacial_area")
        k_f = pp.ad.Scalar(self.fluid.reference_component.thermal_conductivity)
        L = pp.ad.Scalar(self.solid.pore_size)
        return k_f / L * a * 2


class AdditonalTermNakayama:
    def unit_s_fourier_flux(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Discrete Fourier flux on subdomains.

        Parameters:
            subdomains: List of subdomains where the Fourier flux is defined.

        Returns:
            An Ad-operator representing the Fourier flux on the subdomains.

        """
        if len(subdomains) == 0 or isinstance(subdomains[0], pp.BoundaryGrid):
            raise ValueError("Not expected to be called on boundary grids.")

        discr = self.unit_solid_fourier_flux_discretization(subdomains)

        # As opposed to darcy_flux in :class:`DarcyFluxFV`, the gravity term is not
        # included here.
        boundary_operator_fourier = self._combine_boundary_operators(
            subdomains=subdomains,
            dirichlet_operator=self.solid_temperature,
            neumann_operator=self.solid_fourier_flux,
            robin_operator=None,
            bc_type=self.bc_type_solid_fourier_flux,
            name="bc_values_" + self.bc_data_solid_fourier_flux_key,
        )
        flux: pp.ad.Operator = (
            discr.flux() @ self.solid_temperature(subdomains)
            + discr.bound_flux() @ boundary_operator_fourier
        )
        flux.set_name("unit_solid_Fourier_flux")
        return flux

    def unit_f_fourier_flux(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Discrete Fourier flux on subdomains.

        Parameters:
            subdomains: List of subdomains where the Fourier flux is defined.

        Returns:
            An Ad-operator representing the Fourier flux on the subdomains.

        """
        if len(subdomains) == 0 or isinstance(subdomains[0], pp.BoundaryGrid):
            # Given Neumann data prescribed for Fourier flux on boundary.
            raise ValueError("Not expected to be called for boundary grids.")

        discr = self.unit_fluid_fourier_flux_discretization(subdomains)

        # As opposed to darcy_flux in :class:`DarcyFluxFV`, the gravity term is not
        # included here.
        boundary_operator_fourier = self._combine_boundary_operators(
            subdomains=subdomains,
            dirichlet_operator=self.fluid_temperature,
            neumann_operator=self.fourier_flux,
            robin_operator=None,
            bc_type=self.bc_type_fourier_flux,  # This is the fluid BC from the parent
            name="bc_values_" + self.bc_data_fourier_flux_key,
        )
        flux: pp.ad.Operator = (
            discr.flux() @ self.fluid_temperature(subdomains)
            + discr.bound_flux() @ boundary_operator_fourier
        )
        flux.set_name("unit_fluid_Fourier_flux")
        return flux

    def unit_fluid_fourier_flux_discretization(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.MpfaAd:
        """Fourier flux discretization.

        Need fluid and solid to access their separate BCs, even if the thermal conductivity
        is the same (=1).

        Parameters:
            subdomains: List of subdomains where the Fourier flux is defined.

        Returns:
            Discretization object for the Fourier flux.

        """
        return pp.ad.MpfaAd(self.unit_fourier_keyword_fluid, subdomains)

    def unit_solid_fourier_flux_discretization(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.MpfaAd:
        """Fourier flux discretization.

        Parameters:
            subdomains: List of subdomains where the Fourier flux is defined.

        Returns:
            Discretization object for the Fourier flux.

        """
        return pp.ad.MpfaAd(self.unit_fourier_keyword_solid, subdomains)

    def unit_thermal_conductivity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Thermal conductivity [m^2].

        Parameters:
            subdomains: List of subdomains where the thermal conductivity is defined.

        Returns:
            Cell-wise conductivity operator.

        """
        size = sum([sd.num_cells for sd in subdomains])
        # phi = phi * pp.wrap_as_dense_ad_array(1, size)
        conductivity = pp.wrap_as_dense_ad_array(
            1.0,
            size,
            "unit_thermal_conductivity",
        )
        return self.isotropic_second_order_tensor(subdomains, conductivity)

    def set_discretization_parameters(self) -> None:
        """Set default (unitary/zero) parameters for the energy problem.

        The parameter fields of the data dictionaries are updated for all subdomains and
        interfaces (of codimension 1).
        """
        super().set_discretization_parameters()
        for sd, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(
                sd,
                data,
                self.unit_fourier_keyword_solid,
                {
                    "second_order_tensor": self.operator_to_SecondOrderTensor(
                        sd,
                        self.unit_thermal_conductivity([sd]),
                        1.0,
                    ),
                    "ambient_dimension": self.nd,
                    "bc": self.bc_type_solid_fourier_flux(sd),
                },
            )
            pp.initialize_data(
                sd,
                data,
                self.unit_fourier_keyword_fluid,
                {
                    "second_order_tensor": self.operator_to_SecondOrderTensor(
                        sd,
                        self.unit_thermal_conductivity([sd]),
                        1.0,
                    ),
                    "ambient_dimension": self.nd,
                    "bc": self.bc_type_fourier_flux(sd),
                },
            )

    def fluid_solid_energy_exchange(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Energy exchanged from fluid to solid phase.


        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the exchange term.

        """
        op = super().gradient_energy_exchange(subdomains)
        G = self._turtuoisity_G(subdomains)
        k_s = pp.ad.Scalar(self.solid.thermal_conductivity)
        div = pp.ad.Divergence(subdomains)
        op += (
            G
            * k_s
            * (
                div
                @ (
                    self.unit_f_fourier_flux(subdomains)
                    - self.unit_s_fourier_flux(subdomains)
                )
            )
        )
        return op


class ThermalParametersNuskeNoFlow:
    r"""Porosity weighted thermal conductivity.

    Interface coefficient is

    .. math::
        h = k_f / L * a

    where :math:`k_f` is the fluid thermal conductivity, :math:`L` is the pore size, and
    :math:`a` is the fluid-solid interface area.
    """

    def fluid_thermal_conductivity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Thermal conductivity [m^2].

        Parameters:
            subdomains: List of subdomains where the thermal conductivity is defined.

        Returns:
            Cell-wise conductivity operator.

        """

        phi = self._porosity_as_array(subdomains)
        kappa = pp.ad.Scalar(self.fluid.reference_component.thermal_conductivity)
        conductivity = phi * kappa
        conductivity.set_name("fluid_thermal_conductivity")
        return self.isotropic_second_order_tensor(subdomains, conductivity)

    def solid_thermal_conductivity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Thermal conductivity [m^2].

        Parameters:
            subdomains: List of subdomains where the thermal conductivity is defined.

        Returns:
            Cell-wise conductivity operator.

        """
        phi = self._porosity_as_array(subdomains)
        conductivity = (pp.ad.Scalar(1.0) - phi) * pp.ad.Scalar(
            self.solid.thermal_conductivity
        )
        conductivity.set_name("solid_thermal_conductivity")
        return self.isotropic_second_order_tensor(subdomains, conductivity)

    def energy_exchange_coefficient(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.DenseArray:
        """Energy exchange coefficient.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the energy exchange coefficient.

        """
        a = pp.ad.Scalar(self.solid.fluid_solid_interfacial_area)
        a.set_name("fluid_solid_interfacial_area")
        k = self.effective_thermal_conductivity(subdomains)
        L = pp.ad.Scalar(self.solid.pore_size)
        return k / L * a

    def effective_thermal_conductivity(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Effective thermal conductivity [W/m/K].

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the effective thermal conductivity.

        """
        k_f = pp.ad.Scalar(self.fluid.reference_component.thermal_conductivity)
        k_s = pp.ad.Scalar(self.solid.thermal_conductivity)
        return (pp.ad.Scalar(2) * k_f * k_s) / (k_f + k_s)

    def prandtl_number(self, subdomains: list[pp.Grid]) -> pp.ad.Scalar:
        """Prandtl number [-].

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the Prandtl number.

        """
        # https://en.wikipedia.org/wiki/Prandtl_number
        fl = self.fluid.reference_component
        return pp.ad.Scalar(
            fl.specific_heat_capacity * fl.viscosity / fl.thermal_conductivity
        )

    def reynolds_number(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Reynolds number [-].

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the Reynolds number.

        """
        # https://en.wikipedia.org/wiki/Reynolds_number
        f_abs = pp.ad.Function(pp.ad.functions.abs, "velocity magnitude")
        face_velocity = f_abs(self.fluid_flux(subdomains))
        mat = []
        face_areas = []
        for sd in subdomains:
            f2c = np.abs(pp.fvutils.scalar_divergence(sd))
            # Find number of faces for each cell
            n_faces = np.sum(f2c != 0, axis=1)
            # Divide by number of faces
            w = 1 / n_faces
            # Append f2c with weights
            mat.append(sps.diags(w.A.ravel(), 0) * f2c)
            # w = 1 / np.sum(f2c, axis=1)
            # mat.append(f2c * sps.diags(w[:, 0]))
            face_areas.append(sd.face_areas)
        # average_weight = pp.ad.DenseArray(np.concatenate(weight), "average_weight")
        # area and mvem function
        face_to_cell = pp.ad.SparseArray(sps.block_diag(mat), "face_to_cell")
        face_areas = pp.ad.DenseArray(np.concatenate(face_areas), "face_areas")
        mu = pp.ad.Scalar(self.fluid.reference_component.viscosity)
        cell_velocity = face_to_cell @ (face_velocity / face_areas / mu)

        return cell_velocity * pp.ad.Scalar(self.solid.pore_size)

    def nusselt_number(self, subdomains: list[pp.Grid]) -> pp.ad.Scalar:
        """Nusselt number [-].

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the Nusselt number.

        """
        # https://en.wikipedia.org/wiki/Nusselt_number
        nusselt = pp.ad.Scalar(2) + pp.ad.Scalar(1.1) * (
            self.reynolds_number(subdomains)  # ** pp.ad.Scalar(0.6)
        ) * (self.prandtl_number(subdomains) ** pp.ad.Scalar(1 / 3))
        nusselt.set_name("Nusselt_number")
        return nusselt


class ThermalParametersUpscaling(pp.constitutive_laws.ThermalConductivityLTE):
    """Thermal conductivity in the local thermal dynamics.

    Interacts with the rest of the model through calls in set_discretization_parameters.
    """

    solid: LTDSolid

    fluid: LTDFluid

    porosity: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Porosity of the rock. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.ConstantPorosity` or a subclass thereof.

    """
    reference_porosity: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Reference porosity of the rock. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.PoroMechanicsPorosity`.

    """

    def fluid_thermal_conductivity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Thermal conductivity [m^2].

        Parameters:
            subdomains: List of subdomains where the thermal conductivity is defined.

        Returns:
            Cell-wise conductivity operator.

        """
        size = sum([sd.num_cells for sd in subdomains])
        conductivity = pp.wrap_as_dense_ad_array(
            self.fluid.reference_component.specific_pore_contact_area
            * self.fluid.reference_component.thermal_conductivity,
            size,
            "fluid_thermal_conductivity",
        )
        return self.isotropic_second_order_tensor(subdomains, conductivity)

    def solid_thermal_conductivity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Thermal conductivity [m^2].

        Parameters:
            subdomains: List of subdomains where the thermal conductivity is defined.

        Returns:
            Cell-wise conductivity operator.

        """
        size = sum([sd.num_cells for sd in subdomains])
        conductivity = pp.wrap_as_dense_ad_array(
            self.solid.specific_grain_contact_area * self.solid.thermal_conductivity,
            size,
            "solid_thermal_conductivity",
        )
        return self.isotropic_second_order_tensor(subdomains, conductivity)

    def energy_exchange_coefficient(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.DenseArray:
        """Energy exchange coefficient.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the energy exchange coefficient.

        """
        a = pp.ad.Scalar(self.solid.fluid_solid_interfacial_area)
        a.set_name("fluid_solid_interfacial_area")
        h = pp.ad.Scalar(self.solid.interface_heat_transfer_coefficient)
        h.set_name("heat_transfer_coefficient")
        return h * a


class EnthalpyLTNE(pp.constitutive_laws.EnthalpyFromTemperature):
    def solid_enthalpy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Solid enthalpy [J/kg].

        The enthalpy is computed as a perturbation from a reference temperature as
        .. math::
            h = c_p (T - T_0)

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the solid enthalpy.

        """
        c = self.solid_specific_heat_capacity(subdomains)
        enthalpy = c * self.perturbation_from_reference("temperature", subdomains)
        enthalpy.set_name("solid_enthalpy")
        return enthalpy


class NoSolidDiffusion:
    def solid_fourier_flux(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """No diffusion in solid."""
        return pp.wrap_as_dense_ad_array(0, sum([sd.num_faces for sd in subdomains]))


class ConstitutiveLawsLTNE(
    EnthalpyLTNE,
    FouriersLawLTNE,
    pp.constitutive_laws.DimensionReduction,
    ThermalParametersUpscaling,
    pp.constitutive_laws.FluidDensityFromPressureAndTemperature,
    pp.constitutive_laws.ConstantSolidDensity,
):
    """Mixin class for constitutive laws for the LTD model."""

    pass


class VariablesEnergyBalanceLTD(PorePyModel):
    """
    Creates necessary variables (temperatures, advective and diffusive interface flux)
    and provides getter methods for these and their reference values. Getters construct
    mixed-dimensional variables on the fly, and can be called on any subset of the grids
    where the variable is defined. Setter method (assign_variables), however, must
    create on all grids where the variable is to be used.

    Note:
        Wrapping in class methods and not calling equation_system directly allows for
        easier changes of primary variables. As long as all calls to enthalpy_flux()
        accept Operators as return values, we can in theory add it as a primary variable
        and solved mixed form. Similarly for different formulations of enthalpy instead
        of temperature.

    """

    def create_variables(self) -> None:
        """Assign primary variables to subdomains and interfaces of the
        mixed-dimensional grid.

        """
        self.equation_system.create_variables(
            self.solid_temperature_variable,
            subdomains=self.mdg.subdomains(dim=self.nd),
            tags={"si_units": "K"},
        )
        self.equation_system.create_variables(
            self.fluid_temperature_variable,
            subdomains=self.mdg.subdomains(),
            tags={"si_units": "K"},
        )
        self.equation_system.create_variables(
            self.interface_fourier_flux_variable,
            interfaces=self.mdg.interfaces(codim=1),
            tags={"si_units": "W"},
        )
        self.equation_system.create_variables(
            self.interface_enthalpy_flux_variable,
            interfaces=self.mdg.interfaces(codim=1),
            tags={"si_units": "W"},
        )
        self.equation_system.create_variables(
            self.well_enthalpy_flux_variable,
            interfaces=self.mdg.interfaces(codim=2),
            tags={"si_units": "W"},
        )

    def temperature(self, subdomains: list[pp.Grid]) -> pp.ad.MixedDimensionalVariable:
        """Temperature variable. For now, equate LTE temperature to fluid temperature.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Mixed-dimensional variable representing the temperature.

        """
        return self.fluid_temperature(subdomains)

    def fluid_temperature(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Fluid temperature variable.

        Parameters:
            subdomains: List of subdomains or list of boundary grids.

        Returns:
            Mixed-dimensional variable representing the fluid temperature.

        """
        if len(domains) > 0 and all([isinstance(g, pp.BoundaryGrid) for g in domains]):
            return self.create_boundary_operator(
                name=self.fluid_temperature_variable, domains=domains
            )
        t = self.equation_system.md_variable(self.fluid_temperature_variable, domains)
        return t

    def solid_temperature(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Solid temperature variable.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Mixed-dimensional variable representing the solid temperature.

        """
        if len(domains) > 0 and all([isinstance(g, pp.BoundaryGrid) for g in domains]):
            return self.create_boundary_operator(
                name=self.solid_temperature_variable, domains=domains
            )
        else:
            return self.equation_system.md_variable(
                self.solid_temperature_variable, domains
            )

    def interface_fourier_flux(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.MixedDimensionalVariable:
        """Interface Fourier flux.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Variable representing the interface Fourier flux.

        """
        flux = self.equation_system.md_variable(
            self.interface_fourier_flux_variable, interfaces
        )
        return flux

    def interface_enthalpy_flux(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.MixedDimensionalVariable:
        """Interface enthalpy flux.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Variable representing the interface enthalpy flux.
        """
        flux = self.equation_system.md_variable(
            self.interface_enthalpy_flux_variable, interfaces
        )
        return flux

    def well_enthalpy_flux(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.MixedDimensionalVariable:
        """Well enthalpy flux.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Variable representing the well enthalpy flux.

        """
        flux = self.equation_system.md_variable(
            self.well_enthalpy_flux_variable, interfaces
        )
        return flux


class BoundaryConditionsEnergyBalanceLTD(pp.BoundaryConditionMixin):
    def update_boundary_values_primary_variables(self) -> None:
        """Update boundary values for primary variables."""
        super().update_boundary_values_primary_variables()
        self.update_boundary_values(
            self.fluid_temperature_variable, self.fluid_temperature_bc
        )
        self.update_boundary_values(
            self.solid_temperature_variable, self.solid_temperature_bc
        )


class SolutionStrategyEnergyBalanceLTNE(pp.SolutionStrategy):
    """Solution strategy for the energy balance.

    Parameters:
        params: Parameters for the solution strategy.

    """

    def __init__(self, params: Optional[dict] = None) -> None:
        # Generic solution strategy initialization in pp.SolutionStrategy and specific
        # initialization for the fluid mass balance (variables, discretizations...)
        super().__init__(params)

        # Define the energy balance
        # Variables
        self.fluid_temperature_variable: str = "fluid_temperature"
        """Name of the fluid temperature variable."""
        self.temperature_variable = self.fluid_temperature_variable

        self.solid_temperature_variable: str = "solid_temperature"
        """Name of the solid temperature variable."""

        self.interface_fourier_flux_variable: str = "interface_fourier_flux"
        """Name of the primary variable representing the Fourier flux on the
        interface."""

        self.interface_enthalpy_flux_variable: str = "interface_enthalpy_flux"
        """Name of the primary variable representing the enthalpy flux on the
        interface."""
        self.well_enthalpy_flux_variable: str = "well_enthalpy_flux"
        """Name of the primary variable representing the well enthalpy flux on
        interfaces of codimension two."""

        # Discretization
        self.fluid_fourier_keyword: str = "fluid_fourier_keyword"
        """Keyword for Fourier flux term in the fluid phase.

        Used to access discretization parameters and store discretization matrices.

        """
        self.fourier_keyword = self.fluid_fourier_keyword

        self.solid_fourier_keyword: str = "solid_fourier_keyword"
        """Keyword for Fourier flux term in the solid phase.

        Used to access discretization parameters and store discretization matrices.

        """
        self.enthalpy_keyword: str = "enthalpy_flux_discretization"
        """Keyword for enthalpy flux term.

        Used to access discretization parameters and store discretization matrices.

        """

    def set_discretization_parameters(self) -> None:
        """Set default (unitary/zero) parameters for the energy problem.

        The parameter fields of the data dictionaries are updated for all subdomains and
        interfaces (of codimension 1).
        """
        super().set_discretization_parameters()
        for sd, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(
                sd,
                data,
                self.fluid_fourier_keyword,
                {
                    "bc": self.bc_type_fourier_flux(sd),
                    "second_order_tensor": self.operator_to_SecondOrderTensor(
                        sd,
                        self.fluid_thermal_conductivity([sd]),
                        self.fluid.reference_component.thermal_conductivity
                        * self.fluid.reference_component.specific_pore_contact_area,
                    ),
                    "ambient_dimension": self.nd,
                },
            )
            pp.initialize_data(
                sd,
                data,
                self.solid_fourier_keyword,
                {
                    "bc": self.bc_type_solid_fourier_flux(sd),
                    "second_order_tensor": self.operator_to_SecondOrderTensor(
                        sd,
                        self.solid_thermal_conductivity([sd]),
                        self.solid.thermal_conductivity
                        * self.solid.specific_grain_contact_area,
                    ),
                    "ambient_dimension": self.nd,
                },
            )

            pp.initialize_data(
                sd,
                data,
                self.enthalpy_keyword,
                {
                    "bc": self.bc_type_enthalpy_flux(sd),
                },
            )

    def thermal_conductivity_tensor(
        self, sd: pp.Grid, phase: Literal["fluid", "solid"]
    ) -> pp.SecondOrderTensor:
        """Convert ad conductivity to :class:`~pp.params.tensor.SecondOrderTensor`.

        Override this method if the conductivity is anisotropic.

        Parameters:
            sd: Subdomain for which the conductivity is requested.

        Returns:
            Thermal conductivity tensor.

        """
        conductivity_method = getattr(self, phase + "_thermal_conductivity")
        conductivity_ad = self.specific_volume([sd]) * conductivity_method([sd])
        conductivity = conductivity_ad.value(self.equation_system)
        return pp.SecondOrderTensor(conductivity)

    def initial_condition(self) -> None:
        """Add darcy flux to discretization parameter dictionaries."""
        super().initial_condition()
        for sd, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(
                sd,
                data,
                self.enthalpy_keyword,
                {"darcy_flux": np.zeros(sd.num_faces)},
            )
        for intf, data in self.mdg.interfaces(return_data=True):
            pp.initialize_data(
                intf,
                data,
                self.enthalpy_keyword,
                {"darcy_flux": np.zeros(intf.num_cells)},
            )

    def before_nonlinear_iteration(self):
        """Evaluate Darcy flux (super) and copy to the enthalpy flux keyword, to be used
        in upstream weighting.

        """
        # Update parameters *before* the discretization matrices are re-computed.
        for sd, data in self.mdg.subdomains(return_data=True):
            vals = self.darcy_flux([sd]).value(self.equation_system)
            data[pp.PARAMETERS][self.enthalpy_keyword].update({"darcy_flux": vals})

        for intf, data in self.mdg.interfaces(return_data=True, codim=1):
            vals = self.interface_darcy_flux([intf]).value(self.equation_system)
            data[pp.PARAMETERS][self.enthalpy_keyword].update({"darcy_flux": vals})
        for intf, data in self.mdg.interfaces(return_data=True, codim=2):
            vals = self.well_flux([intf]).value(self.equation_system)
            data[pp.PARAMETERS][self.enthalpy_keyword].update({"darcy_flux": vals})

        super().before_nonlinear_iteration()

    def set_nonlinear_discretizations(self) -> None:
        """Collect discretizations for nonlinear terms."""
        super().set_nonlinear_discretizations()
        self.add_nonlinear_discretization(
            self.enthalpy_discretization(self.mdg.subdomains()).upwind(),
        )
        self.add_nonlinear_discretization(
            self.interface_enthalpy_discretization(self.mdg.interfaces()).flux(),
        )

    def update_boundary_values_primary_variables(self) -> None:
        """Update boundary values for primary variables."""

        self.update_boundary_values(
            self.fluid_temperature_variable, self.fluid_temperature_bc
        )
        self.update_boundary_values(
            self.solid_temperature_variable, self.solid_temperature_bc
        )


class EnergyBalanceLTD(
    EnergyBalanceEquationsLTNE,
    ConstitutiveLawsLTNE,
    VariablesEnergyBalanceLTD,
    SolutionStrategyEnergyBalanceLTNE,
):
    pass


mass = pp.fluid_mass_balance


class VariablesMassAndEnergy(
    VariablesEnergyBalanceLTD,
    mass.VariablesSinglePhaseFlow,
):
    """Combines mass and momentum balance variables."""

    def create_variables(self):
        """Set the variables for the poromechanics problem.

        Call all parent classes' set_variables methods.

        """
        # Energy balance and its parent mass balance
        VariablesEnergyBalanceLTD.create_variables(self)
        mass.VariablesSinglePhaseFlow.create_variables(self)


class EquationsMassAndEnergy(
    EnergyBalanceEquationsLTNE,
    mass.FluidMassBalanceEquations,
):
    """Combines energy, mass and momentum balance equations."""

    def set_equations(self) -> None:
        """Set the equations for the problem.

        Call all parent classes' set_equations methods.

        """
        # Call all super classes' set_equations methods. Do this explicitly (calling the
        # methods of the super classes directly) instead of using super() since this is
        # more transparent.
        EnergyBalanceEquationsLTNE.set_equations(self)
        mass.FluidMassBalanceEquations.set_equations(self)


class SolutionStrategyMassAndEnergy(
    SolutionStrategyEnergyBalanceLTNE,
    mass.SolutionStrategySinglePhaseFlow,
):
    pass


class ConstitutiveLawsMassAndEnergy(
    ConstitutiveLawsLTNE,
    FluxDiscretization,
    mass.ConstitutiveLawsSinglePhaseFlow,
):
    pass


class BoundaryConditionsMassAndEnergy(
    pp.energy_balance.BoundaryConditionsEnergyBalance,
    mass.BoundaryConditionsSinglePhaseFlow,
):
    bc_data_solid_fourier_flux_key: str = "solid_fourier_flux"
    """Keyword for the storage of Neumann-type boundary conditions for the solid Fourier
    flux."""

    def bc_type_solid_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary conditions on all external boundaries for the conductive flux
        in the energy equation.

        Parameters:
            sd: Subdomain grid on which to define boundary conditions.

        Returns:
            Boundary condition object. Per default Dirichlet-type BC are assigned,
            requiring temperature values on the bonudary.

        """
        # Define boundary faces.
        boundary_faces = self.domain_boundary_sides(sd).all_bf
        # Define boundary condition on all boundary faces.
        return pp.BoundaryCondition(sd, boundary_faces, "dir")

    def bc_values_solid_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Temperature values for the Dirichlet boundary condition.

        These values are used for quantities relying on Dirichlet data for temperature
        on the boundary, such as the Fourier flux.

        Important:
            Override this method to provide custom Dirichlet boundary data for
            temperature, per boundary grid as a numpy array with numerical values.

        Parameters:
            boundary_grid: Boundary grid to provide values for.

        Returns:
            An array with ``shape=(boundary_grid.num_cells,)`` containing temperature
            values on the provided boundary grid.

        """
        return self.reference_variable_values.temperature * np.ones(
            boundary_grid.num_cells
        )

    def bc_values_solid_fourier_flux(
        self, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        """**Heat** flux values on the Neumann boundary to be used with Fourier's law.

        The values are used on the boundary for :math:`c \\nabla T` where Neumann data
        is required for the whole expression
        (``c`` being the conductivity on the boundary).

        Important:
            Override this method to provide custom Neumann boundary data for
            the flux, per boundary grid as a numpy array with numerical values.

        Parameters:
            boundary_grids: Boundary grid to provide values for.

        Returns:
            Numeric Fourier flux values for a Neumann-type BC.

        """
        return np.zeros(boundary_grid.num_cells)

    def update_all_boundary_conditions(self) -> None:
        """Set values for the temperature and the Fourier flux on boundaries.

        Note:
            This assumes as of now that Dirichlet-type BC are provided only for
            temperature.
            Work must be done if other energy-related quantities are defined as
            primary variables.

        """
        mass.BoundaryConditionsSinglePhaseFlow.update_all_boundary_conditions(self)

        # Update Neumann conditions
        self.update_boundary_condition(
            name=self.bc_data_fourier_flux_key, function=self.bc_values_fourier_flux
        )
        self.update_boundary_condition(
            name=self.bc_data_solid_fourier_flux_key,
            function=self.bc_values_solid_fourier_flux,
        )
        # Update Dirichlet conditions
        self.update_boundary_condition(
            name=self.fluid_temperature_variable,
            function=self.bc_values_temperature,
        )
        self.update_boundary_condition(
            name=self.solid_temperature_variable,
            function=self.bc_values_solid_temperature,
        )

        self.update_boundary_condition(
            name=self.bc_data_enthalpy_flux_key, function=self.bc_values_enthalpy_flux
        )


class NonzeroInitialCondition(pp.PorePyModel):
    def initial_condition(self) -> None:
        """Set the initial condition for the problem."""
        super().initial_condition()
        for var in self.equation_system.variables:
            values = getattr(self, "ic_values_" + var.name)(var.domain)
            self.equation_system.set_variable_values(
                values, [var], iterate_index=0, time_step_index=0
            )

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        val = self.reference_variable_values.pressure
        return val * np.ones(sd.num_cells)

    def ic_values_solid_temperature(self, sd: pp.Grid) -> np.ndarray:
        val = self.reference_variable_values.temperature
        return val * np.ones(sd.num_cells)


class NonzeroInitialConditionLTNE(NonzeroInitialCondition):
    def ic_values_fluid_temperature(self, sd: pp.Grid) -> np.ndarray:
        val = self.reference_variable_values.temperature
        return val * np.ones(sd.num_cells)

    def ic_values_solid_temperature(self, sd: pp.Grid) -> np.ndarray:
        val = self.reference_variable_values.temperature
        return val * np.ones(sd.num_cells)


class FluidMassAndEnergyBalanceLTNE(
    pp.DataSavingMixin,
    pp.ModelGeometry,
    EquationsMassAndEnergy,
    ConstitutiveLawsMassAndEnergy,
    VariablesMassAndEnergy,
    BoundaryConditionsMassAndEnergy,
    NonzeroInitialConditionLTNE,
    pp.FluidMixin,
    SolutionStrategyMassAndEnergy,
):
    pass
