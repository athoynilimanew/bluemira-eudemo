# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Geometry Functions to make the CSG branch work."""

from copy import deepcopy
from enum import Enum, auto

from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.reactor import Reactor
from bluemira.geometry.tools import revolve_shape
from bluemira.radiation_transport.generalised_neutronics.geometry import (
    NeutronicsGeometryManager,
)
from matproplib.material import Material

from eudemo.blanket import Blanket
from eudemo.comp_managers import ThermalShield
from eudemo.ivc.divertor_silhouette import Divertor
from eudemo.vacuum_vessel import VacuumVessel


class EUDEMOGeometryModel(Enum):
    """
    Enumeration of different geometry model to be
    considered for neutronics simulations
    """

    SIMPLE = auto()

    @classmethod
    def _missing_(cls, value: str):
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(
                f"{cls.__name__} has no type {value}. "
                f"Please select from {(*cls._member_names_,)}"
            ) from None


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


def simplify_vacuum_vessel_tree(vacuum_vessel: VacuumVessel) -> list[Component]:
    """Simplify the VacuumVessel component tree for neutronics calculations.

    Parameters
    ----------
    vacuum_vessel:
        Vacuum vessel component.

    Returns
    -------
    list[Component]
        Simplified vacuum vessel components.

    Note
    -------
    EUDEMO vacuum vessel xz and xyz Tree Structure,
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

    body_mat = next(child.material for child in vv_xyz.children if child.name == "Body")

    void_mat = next(
        child.material for child in vv_xyz.children if child.name == "Vessel voidspace"
    )

    materials = [body_mat, body_mat, body_mat, void_mat]

    return _create_multiple_xz_components(xz_phys_components, materials)


def simplify_thermal_shield_tree(thermal_shield: ThermalShield) -> list[Component]:
    """Simplify the ThermalShield component tree for neutronics calculations.

    Parameters
    ----------
    thermal_shield:
        Thermal shield component.

    Returns
    -------
    list[Component]
        Simplified thermal shield components.

    Note
    -------
    EUDEMO thermal shield xz and xyz Tree Structure,
    as of Oct 2026 for reference

    Thermal Shield (Component)
    ├── VVTS (Component)
    │   ├── xyz (Component)
    │   │   └── Sector 1 (Component)
    │   │       ├── VVTS 1 (PhysicalComponent)
    │   │       └── VVTS voidspace 1 (PhysicalComponent)
    │   ├── xz (Component)
    │   │   ├── VVTS 1 0 (PhysicalComponent)
    │   │   ├── VVTS 1 1 (PhysicalComponent)
    │   │   ├── VVTS 1 2 (PhysicalComponent)
    │   │   └── VVTS voidspace 1 0 (PhysicalComponent)
    └── CryostatTS (Component)
        ├── xyz (Component)
        │   └── Sector 1 (Component)
        │       ├── Cryostat TS 1 (PhysicalComponent)
        │       └── Cryostat voidspace 1 (PhysicalComponent)
        ├── xz (Component)
        │   ├── Cryostat TS 1 0 (PhysicalComponent)
        │   ├── Cryostat TS 1 1 (PhysicalComponent)
        │   ├── Cryostat TS 1 2 (PhysicalComponent)
        │   ├── Cryostat TS 1 3 (PhysicalComponent)
        │   └── Cryostat voidspace 1 0 (PhysicalComponent)

    The xyz representation is not for full 365 degrees
    so need to recreate
    """
    xz_phys_components = [
        *thermal_shield.component().get_component("VVTS").get_component("xz").children,
        *thermal_shield
        .component()
        .get_component("CryostatTS")
        .get_component("xz")
        .children,
    ]

    # TODO: Remove the hard codes and create a cleaner way to do the same
    vvts_xyz = (
        thermal_shield
        .component()
        .get_component("VVTS")
        .get_component("xyz")
        .get_component("Sector 1")
    )

    vvts_mat = next(
        child.material for child in vvts_xyz.children if child.name == "VVTS 1"
    )
    void_mat = next(
        child.material for child in vvts_xyz.children if child.name == "VVTS voidspace 1"
    )

    cryo_xyz = (
        thermal_shield
        .component()
        .get_component("CryostatTS")
        .get_component("xyz")
        .get_component("Sector 1")
    )

    ts_mat = next(
        child.material for child in cryo_xyz.children if child.name == "Cryostat TS 1"
    )
    cryo_void_mat = next(
        child.material
        for child in cryo_xyz.children
        if child.name == "Cryostat voidspace 1"
    )

    materials = [
        vvts_mat,
        vvts_mat,
        vvts_mat,
        void_mat,
        ts_mat,
        ts_mat,
        ts_mat,
        ts_mat,
        cryo_void_mat,
    ]
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


def get_all_simplified_components(
    blanket: Blanket,
    vacuum_vessel: VacuumVessel,
    thermal_shield: ThermalShield,
    divertor: Divertor,
) -> list[Component]:
    """Get all simplified components for neutronics calculations.

    Parameters
    ----------
    blanket:
        Blanket component.
    vacuum_vessel:
        Vacuum vessel component.
    thermal_shield:
        Thermal shield component.
    divertor:
        Divertor component.

    Returns
    -------
    list[Component]
        All simplified components.
    """
    # TODO: Provide appropriate Material Mapping

    return [
        *simplify_blanket_tree(blanket),
        *simplify_vacuum_vessel_tree(vacuum_vessel),
        *simplify_thermal_shield_tree(thermal_shield),
        *simplify_divertor_tree(divertor),
    ]


def despline_reactor(
    reactor: Reactor, geom_model: EUDEMOGeometryModel = EUDEMOGeometryModel.SIMPLE
) -> NeutronicsGeometryManager:
    """Despline the geometry of an EUDEMO reactor.

    Parameters
    ----------
    reactor
        EUDEMO reactor containing the components to despline.
    geom_model
        Geometry model used to obtain the component managers.

    Returns
    -------
    NeutronicsGeometryManager
        Desplined reactor geometry.
    """
    if geom_model == EUDEMOGeometryModel.SIMPLE:
        all_comps = get_all_simplified_components(
            blanket=reactor.blanket,
            vacuum_vessel=reactor.vacuum_vessel,
            thermal_shield=reactor.thermal_shield,
            divertor=reactor.divertor,
        )
    else:
        raise NotImplementedError

    discretisations = len(all_comps) * [10]

    return NeutronicsGeometryManager.from_list_of_components(
        components=all_comps,
        discretisations=discretisations,
        overlap_tolerance=1e-10,
        gap_tolerance=1e-6,
    )
