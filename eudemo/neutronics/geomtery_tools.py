# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Geometry Functions to make the CSG branch work."""

from bluemira.base.components import Component, PhysicalComponent
from bluemira.geometry.tools import revolve_shape
from matproplib.material import Material

from eudemo.blanket import Blanket
from eudemo.comp_managers import ThermalShield
from eudemo.ivc.divertor_silhouette import Divertor
from eudemo.vacuum_vessel import VacuumVessel


def _simplify_xz_components(
    xz_components: list[PhysicalComponent],
    material: Material | None = None,
) -> list[Component]:
    """Create simplified xz/xyz component pairs.

    Parameters
    ----------
    xz_components:
        Physical components from the xz tree.
    material:
        Material assigned to the revolved components.

    Returns
    -------
    list[Component]
        Simplified components for neutronics calculations.
    """
    components = []

    for xz_phys in xz_components:
        revolved_body = PhysicalComponent(
            name=xz_phys.name,
            shape=revolve_shape(
                xz_phys.shape,
                base=(0, 0, 0),
                direction=(0, 0, 1),
                degree=360.0,
            ),
            material=material,
        )
        components.append(
            Component(
                name=xz_phys.name,
                children=[
                    Component("xz", children=[xz_phys]),
                    Component("xyz", children=[revolved_body]),
                ],
            )
        )

    return components


def simplify_blanket_tree(
    blanket: Blanket, material: Material | None = None
) -> list[Component]:
    """Simplify the blanket component tree for neutronics calculations.

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
    """
    xz_components = blanket.component().get_component("xz").get_component("BB").children
    return _simplify_xz_components(xz_components, material)


def simplify_vacuum_vessel_tree(
    vv: VacuumVessel, material: Material | None = None
) -> list[Component]:
    """Simplify the VacuumVessel component tree for neutronics calculations.

    Parameters
    ----------
    vv:
        Vacuum vessel component.
    material:
        Material assigned to the revolved components.

    Returns
    -------
    list[Component]
        Simplified vacuum vessel components.
    """
    xz_components = vv.component().get_component("xz").children

    for xz_phys in xz_components:
        xz_phys.name = f"VacuumVessel_{xz_phys.name}"

    return _simplify_xz_components(xz_components, material)


def simplify_thermal_shield_tree(
    ts: ThermalShield, material: Material | None = None
) -> list[Component]:
    """Simplify the ThermalShield component tree for neutronics calculations.

    Parameters
    ----------
    ts:
        Thermal shield component.
    material:
        Material assigned to the revolved components.

    Returns
    -------
    list[Component]
        Simplified thermal shield components.
    """
    xz_components = [
        *ts.component().get_component("VVTS").get_component("xz").children,
        *ts.component().get_component("CryostatTS").get_component("xz").children,
    ]
    return _simplify_xz_components(xz_components, material)


def simplify_divertor_tree(
    div: Divertor, material: Material | None = None
) -> list[Component]:
    """Simplify the Divertor component tree for neutronics calculations.

    Parameters
    ----------
    div:
        Divertor component.
    material:
        Material assigned to the revolved components.

    Returns
    -------
    list[Component]
        Simplified divertor components.
    """
    xz_components = div.component().get_component("xz").children
    return _simplify_xz_components(xz_components, material)
