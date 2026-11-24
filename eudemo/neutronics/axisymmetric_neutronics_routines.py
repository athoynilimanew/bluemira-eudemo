# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Geometry Functions to make the CSG branch work."""

from copy import deepcopy

from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.look_and_feel import bluemira_print
from bluemira.base.reactor import Reactor
from bluemira.geometry.tools import revolve_shape
from bluemira.radiation_transport.axisymmetric_neutronics.neutronics_reactor import (
    AxisymmetricNeutronicsReactor,
    NeutronicsModel,
)
from matproplib.material import Material

from eudemo.blanket import Blanket
from eudemo.comp_managers import RadiationShield, ThermalShield
from eudemo.ivc.divertor_silhouette import Divertor
from eudemo.vacuum_vessel import VacuumVessel


def _create_multiple_xz_components(
    xz_phys_components: list[PhysicalComponent],
    materials: list[Material] | None = None,
) -> list[Component]:
    """Create simplified xz/xyz component pairs.

    Parameters
    ----------
    xz_phys_components:
        Physical components from the xz tree.
    materials:
        Material assigned to the revolved components.

    Returns
    -------
    list[Component]
        Simplified components for neutronics calculations.
    """
    components = []

    for xz_phys, mat in zip(xz_phys_components, materials, strict=True):
        # To avoid original component mutation
        copied_component = deepcopy(xz_phys)
        revolved_body = PhysicalComponent(
            name=copied_component.name,
            shape=revolve_shape(
                copied_component.shape,
                base=(0, 0, 0),
                direction=(0, 0, 1),
                degree=360.0,
            ),
            material=mat,
        )
        components.append(
            Component(
                name=copied_component.name,
                children=[
                    Component("xz", children=[copied_component]),
                    Component("xyz", children=[revolved_body]),
                ],
            )
        )

    return components


def simplify_blanket_tree(blanket: Blanket) -> list[Component]:
    """
    Simplify the blanket component tree for neutronics calculations.

    Parameters
    ----------
    blanket:
        Blanket component.
    material:
        Material assigned to the revolved components.

    Returns
    -------
    list[Component]
        Simplified blanket components.

    Note
    -------
    EUDEMO Blanket xz and xyz Tree Structure,
    as of Oct 2026 for reference

    Blanket (Component)
    ├── xz (Component)
    │   └── BB (Component)
    │       ├── IBS_FW (PhysicalComponent)
    │       ├── IBS_BZ (PhysicalComponent)
    │       ├── IBS_MANIFOLD (PhysicalComponent)
    │       ├── OBS_FW (PhysicalComponent)
    │       ├── OBS_BZ (PhysicalComponent)
    │       └── OBS_MANIFOLD (PhysicalComponent)
    └── xyz (Component)
        └── Sector 1 (Component)
            └── BB 1 (Component)
                ├── IBS_FW_0 1 (PhysicalComponent)
                ├── IBS_FW_1 1 (PhysicalComponent)
                ├── IBS_BZ_0 1 (PhysicalComponent)
                ├── IBS_BZ_1 1 (PhysicalComponent)
                ├── IBS_MANIFOLD_0 1 (PhysicalComponent)
                ├── IBS_MANIFOLD_1 1 (PhysicalComponent)
                ├── OBS_FW_0 1 (PhysicalComponent)
                ├── OBS_FW_1 1 (PhysicalComponent)
                ├── OBS_FW_2 1 (PhysicalComponent)
                ├── OBS_BZ_0 1 (PhysicalComponent)
                ├── OBS_BZ_1 1 (PhysicalComponent)
                ├── OBS_BZ_2 1 (PhysicalComponent)
                ├── OBS_MANIFOLD_0 1 (PhysicalComponent)
                ├── OBS_MANIFOLD_1 1 (PhysicalComponent)
                └── OBS_MANIFOLD_2 1 (PhysicalComponent)

    For neutronics calculations, we want a simpler XYZ representation,
    hence this function. Also, The xyz representation is not for
    full 365 degrees so need to recreate.
    """
    xz_phys_components = (
        blanket.component().get_component("xz").get_component("BB").children
    )

    # TODO: Remove the hard codes and create a cleaner way to do the same
    bb_1_xyz = (
        blanket
        .component()
        .get_component("xyz")
        .get_component("Sector 1")
        .get_component("BB 1")
    )

    IBS_FW_mat = next(
        child.material for child in bb_1_xyz.children if child.name == "IBS_FW_0 1"
    )
    IBS_BZ_mat = next(
        child.material for child in bb_1_xyz.children if child.name == "IBS_BZ_0 1"
    )
    IBS_MANIFOLD_mat = next(
        child.material for child in bb_1_xyz.children if child.name == "IBS_MANIFOLD_0 1"
    )
    OBS_FW_mat = next(
        child.material for child in bb_1_xyz.children if child.name == "OBS_FW_0 1"
    )
    OBS_BZ_mat = next(
        child.material for child in bb_1_xyz.children if child.name == "OBS_BZ_0 1"
    )
    OBS_MANIFOLD_mat = next(
        child.material for child in bb_1_xyz.children if child.name == "OBS_MANIFOLD_0 1"
    )

    materials = [
        IBS_FW_mat,
        IBS_BZ_mat,
        IBS_MANIFOLD_mat,
        OBS_FW_mat,
        OBS_BZ_mat,
        OBS_MANIFOLD_mat,
    ]

    return _create_multiple_xz_components(xz_phys_components, materials)


def simplify_vacuum_vessel_tree(
    vacuum_vessel: VacuumVessel, *, with_ports: bool = False
) -> list[Component]:
    """Simplify the VacuumVessel component tree for neutronics calculations.

    Parameters
    ----------
    vacuum_vessel:
        Vacuum vessel component.

    Returns
    -------
    :
        Simplified vacuum vessel components.

    Note
    -------
    EUDEMO vacuum vessel xz and xyz Tree Structure, (with ports)
    as of Oct 2026 for reference

    VacuumVessel (Component)
    ├── xyz (Component)
    │   └── Sector 1 (Component)
    │       ├── Body (PhysicalComponent)
    │       └── Vessel voidspace (PhysicalComponent)
    ├── xz (Component)
    │   ├── Body 0 (PhysicalComponent)
    │   ├── Body 1 (PhysicalComponent)
    │   ├── Body 2 (PhysicalComponent)
    │   └── Vessel voidspace 0 (PhysicalComponent)

    The xyz representation is not for full 365 degrees
    so need to recreate
    """
    xz_phys_components = vacuum_vessel.component().get_component("xz").children

    for xz_phys in xz_phys_components:
        xz_phys.name = f"VacuumVessel_{xz_phys.name}"

    # TODO: Remove the hard codes and create a cleaner way to do the same
    vv_xyz = vacuum_vessel.component().get_component("xyz").get_component("Sector 1")

    if with_ports:
        body_mat_0 = next(
            child.material for child in vv_xyz.children if child.name == "Body 0"
        )
        body_mat_1 = next(
            child.material for child in vv_xyz.children if child.name == "Body 1"
        )
        body_mat_2 = next(
            child.material for child in vv_xyz.children if child.name == "Body 2"
        )

        void_mat = next(
            child.material
            for child in vv_xyz.children
            if child.name == "Vessel voidspace 0"
        )

        materials = [body_mat_0, body_mat_1, body_mat_2, void_mat]

    else:
        body_mat = next(
            child.material for child in vv_xyz.children if child.name == "Body 1"
        )

        void_mat = next(
            child.material
            for child in vv_xyz.children
            if child.name == "Vessel voidspace 1"
        )

        materials = [body_mat, void_mat]

    return _create_multiple_xz_components(xz_phys_components, materials)


def simplify_radiation_shield_tree(
    radiation_shield: RadiationShield, *, with_plugs: bool = False
) -> list[Component]:
    """Simplify the RadiationShield component tree for neutronics calculations.

    Parameters
    ----------
    radiation_shield:
        Radiation shield component.
    with_plugs:
        Whether to include plugs in the radiation shield.

    Returns
    -------
    list[Component]
        Simplified radiation shield components.

    Note
    -------
    EUDEMO radiation shield xz, xy and xyz Tree Structure,
    as of Oct 2026 for reference. (withput plugs)

    RadiationShield (Component)
    ├── xz (Component)
    │   └── Body (PhysicalComponent)
    ├── xy (Component)
    │   └── Body (PhysicalComponent)
    └── xyz (Component)
        └── Sector 1 (Component)
            └── Body 1 (PhysicalComponent)

    The xyz representation is not for full 360 degrees
    so need to recreate

    Raises
    ------
    NotImplementedError
        if with_plugs = True
    """
    if with_plugs:
        raise NotImplementedError("Radiation shield with plugs is not yet implemented.")

    xz_phys_components = radiation_shield.component().get_component("xz").children

    for xz_phys in xz_phys_components:
        xz_phys.name = f"RadiationShield_{xz_phys.name}"

    # TODO: Remove the hard codes and create a cleaner way to do the same.
    rs_xyz = radiation_shield.component().get_component("xyz").get_component("Sector 1")

    body_mat = next(
        child.material for child in rs_xyz.children if child.name == "Body 1"
    )

    materials = [body_mat]

    return _create_multiple_xz_components(xz_phys_components, materials)


def simplify_thermal_shield_tree(
    thermal_shield: ThermalShield, *, with_ports: bool = False
) -> list[Component]:
    """Simplify the ThermalShield component tree for neutronics calculations.

    Parameters
    ----------
    thermal_shield:
        Thermal shield component.
    with_ports:
        Whether to include port-related material subdivisions.

    Returns
    -------
    list[Component]
        Simplified thermal shield components.

    Note
    -------    EUDEMO thermal shield xz and xyz tree structure,
    as of Oct 2026.

    The xyz representation is not for full 360 degrees,
    so it needs to be recreated.
    """
    vvts = thermal_shield.component().get_component("VVTS")
    cryostat_ts = thermal_shield.component().get_component("CryostatTS")

    xz_phys_components = [
        *vvts.get_component("xz").children,
        *cryostat_ts.get_component("xz").children,
    ]

    # Get materials from the corresponding xyz representations.
    vvts_xyz = vvts.get_component("xyz").get_component("Sector 1")
    cryostat_xyz = cryostat_ts.get_component("xyz").get_component("Sector 1")

    vvts_materials = {child.name: child.material for child in vvts_xyz.children}
    cryostat_materials = {child.name: child.material for child in cryostat_xyz.children}

    if with_ports:
        materials = [
            vvts_materials["VVTS 1"],
            vvts_materials["VVTS 1"],
            vvts_materials["VVTS 1"],
            vvts_materials["VVTS voidspace 1"],
            cryostat_materials["Cryostat TS 1"],
            cryostat_materials["Cryostat TS 1"],
            cryostat_materials["Cryostat TS 1"],
            cryostat_materials["Cryostat TS 1"],
            cryostat_materials["Cryostat voidspace 1"],
        ]
    else:
        materials = [
            vvts_materials["VVTS 1"],
            vvts_materials["VVTS voidspace 1"],
            cryostat_materials["Cryostat TS 1"],
            cryostat_materials["Cryostat voidspace 1"],
        ]

    for component in xz_phys_components:
        component.name = f"ThermalShield_{component.name}"

    return _create_multiple_xz_components(xz_phys_components, materials)


def simplify_divertor_tree(divertor: Divertor) -> list[Component]:
    """
    Simplify the Divertor component tree for neutronics calculations.

    Parameters
    ----------
    div:
        Divertor component.
    Returns
    -------
    list[Component]
        Simplified divertor components.

    Note
    ------
    EUDEMO divertor xz and xyz Tree Structure,
    as of Oct 2026 for reference

    Divertor (Component)
    ├── xz (Component)
    │   └── Body (PhysicalComponent)
    └── xyz (Component)
        └── Sector 1 (Component)
            └── cassettes 1 (Component)
                ├── segment_0 1 (PhysicalComponent)
                ├── segment_1 1 (PhysicalComponent)
                └── segment_2 1 (PhysicalComponent)

    For neutronics calculations, we want a simpler XYZ representation,
    hence this function. Also, The xyz representation is not for
    full 365 degrees so need to recreate.
    """
    xz_phys_components = divertor.component().get_component("xz").children
    for xz_phys in xz_phys_components:
        xz_phys.name = f"Divertor_{xz_phys.name}"

    materials = [
        divertor
        .component()
        .get_component("xyz")
        .get_component("Sector 1")
        .get_component("cassettes 1")
        .children[0]
        .material
    ]
    return _create_multiple_xz_components(xz_phys_components, materials)


class EUDEMONeutronicsReactor(AxisymmetricNeutronicsReactor):
    """EUDEMO Axis-symmetric neutronics model"""

    def get_all_components(self, reactor: Reactor) -> list[Component]:
        """
        Get all relevant components for the reactor

        Returns
        -------
        list
            list of components.
        """

        if self.model == NeutronicsModel.SIMPLE:
            return [
                *simplify_blanket_tree(reactor.blanket),
                *simplify_divertor_tree(reactor.divertor),
                *simplify_vacuum_vessel_tree(reactor.vacuum_vessel, with_ports=False),
            ]
        if self.model == NeutronicsModel.ADVANCED:
            return [
                *simplify_blanket_tree(reactor.blanket),
                *simplify_divertor_tree(reactor.divertor),
                *simplify_vacuum_vessel_tree(reactor.vacuum_vessel, with_ports=False),
                *simplify_thermal_shield_tree(reactor.thermal_shield, with_ports=False),
                *simplify_radiation_shield_tree(
                    reactor.radiation_shield, with_plugs=False
                ),
            ]
        raise NotImplementedError

    @staticmethod
    def default_discretisations(comp_name: str) -> int:
        """
        Return the default spline discretisation for a component.

        Returns
        -------
        int
        """
        # TODO: remove this hardcoded constant discretisation
        bluemira_print(f"Using a constant discretisation for{comp_name}")
        return 100

    def customised_tallies(self, model):
        """Optionally customise the model tallies."""
        pass
