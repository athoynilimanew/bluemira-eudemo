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


def simplify_blanket_tree(blanket: Blanket, material: Material) -> list[Component]:
    """
    Simplify the current component tree of the blanket
    for neutronics calculations.




    """

    blanket_components = []

    all_xz_phys = blanket.component().get_component("xz").get_component("BB").children

    for xz_phys in all_xz_phys:
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

        blanket_components.append(
            Component(
                name=xz_phys.name,
                children=[
                    Component(
                        "xz",
                        children=[xz_phys],
                    ),
                    Component(
                        "xyz",
                        children=[revolved_body],
                    ),
                ],
            )
        )

    return blanket_components
