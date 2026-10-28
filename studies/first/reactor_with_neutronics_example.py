# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Minimum Viable Example of Axissymmetric Neutronics
for the EUDEMO reactor design.

1. Radial build (using PROCESS)
2. Perform equilibria optimisation
3. Build plasma
4. Design scaffold for the IVCs
5. Build vacuum vessel
6. Build TF coils
7. Build PF coils
8. Build cryo thermal shield
9. Build cryostat
10. Build radiation shield
11. Produce power cycle report
"""

from pathlib import Path

from bluemira.base.designer import run_designer
from bluemira.base.logs import set_log_level
from bluemira.base.look_and_feel import bluemira_error
from bluemira.base.reactor_config import ReactorConfig
from bluemira.materials.cache import establish_material_cache
from bluemira.radiation_transport.axisymmetric_neutronics.neutronics_reactor import (
    NeutronicsModel,
)
from matproplib.conditions import OperationalConditions
from reactor import (
    EUDEMO,
    build_blanket,
    build_divertor,
    build_plasma,
    build_reference_equilibrium,
    build_vacuum_vessel,
)

from eudemo.equilibria import (
    DummyFixedEquilibriumDesigner,
    FixedEquilibriumDesigner,
)
from eudemo.ivc import design_ivc
from eudemo.maintenance.upper_port import UpperPortKOZDesigner
from eudemo.model_managers import EquilibriumManager
from eudemo.neutronics.geomtery_tools import EUDEMONeutronicsReactor
from eudemo.params import EUDEMOReactorParams

STUDY = Path(__file__).parent
CONFIG_DIR = STUDY / "config"
BUILD_CONFIG_FILE_PATH = Path(CONFIG_DIR, "build_config.json").as_posix()
VERSION = 5

if __name__ == "__main__":
    set_log_level("INFO")
    import time

    run_time_track = {
        "Total": 0.0,
        "CSG neutronics": 0.0,
    }

    try:  # noqa: PLW0717
        start = time.time()
        reactor_config = ReactorConfig(BUILD_CONFIG_FILE_PATH, EUDEMOReactorParams)
        reactor = EUDEMO("EUDEMO", n_sectors=reactor_config.global_params.n_TF.value)

        establish_material_cache([
            "eudemo.materials",
            "eurofusion_materials.library",
            "matproplib",
        ])

        lcfs_coords, profiles = run_designer(
            FixedEquilibriumDesigner,
            reactor_config.params_for("Fixed boundary equilibrium"),
            reactor_config.config_for("Fixed boundary equilibrium"),
        )

        lcfs_coords, profiles = run_designer(
            DummyFixedEquilibriumDesigner,
            reactor_config.params_for("Dummy fixed boundary equilibrium"),
            reactor_config.config_for("Dummy fixed boundary equilibrium"),
        )

        reactor.equilibria = EquilibriumManager()

        reference_eq = build_reference_equilibrium(
            reactor_config.params_for("Free boundary equilibrium"),
            reactor_config.config_for("Free boundary equilibrium"),
            reactor.equilibria,
            lcfs_coords,
            profiles,
        )

        reactor.plasma = build_plasma(
            reactor_config.params_for("Plasma"),
            reactor_config.config_for("Plasma"),
            reference_eq,
        )

        ivc_shapes = design_ivc(
            reactor_config.params_for("IVC").global_params,
            reactor_config.config_for("IVC"),
            equilibrium=reference_eq,
        )

        reactor.vacuum_vessel = build_vacuum_vessel(
            reactor_config.params_for("Vacuum vessel"),
            reactor_config.config_for("Vacuum vessel"),
            ivc_shapes.outer_boundary,
        )

        reactor.divertor = build_divertor(
            reactor_config.params_for("Divertor"),
            reactor_config.config_for("Divertor"),
            ivc_shapes.divertor_face,
        )

        upper_port_designer = UpperPortKOZDesigner(
            reactor_config.params_for("Upper Port"),
            reactor_config.config_for("Upper Port"),
            ivc_shapes.blanket_face,
        )
        upper_port_koz_xz, r_inner_cut, cut_angle = upper_port_designer.execute()

        reactor.blanket = build_blanket(
            reactor_config.params_for("Blanket"),
            reactor_config.config_for("Blanket"),
            ivc_shapes.inner_boundary,
            ivc_shapes.blanket_face,
            r_inner_cut,
            cut_angle,
        )

        neutronics_start = time.time()
        csg_reactor = EUDEMONeutronicsReactor(
            neutronics_config=reactor_config.config_for("Neutronics").get("CSG"),
            params=reactor_config.params_for("Neutronics"),
            model=NeutronicsModel.SIMPLE,
            reactor=reactor,
        )

        csg_reactor.geometry.plot("xz")
        csg_reactor.run_mcio(
            op_cond=OperationalConditions(temperature=298, pressure=101325)
        )
        neutronics_end = time.time()
        run_time_track["CSG neutronics"] = neutronics_end - neutronics_start

        end = time.time()
        run_time_track["Total"] = end - start

    except Exception as e:
        bluemira_error(e.with_traceback(e.__traceback__))
        raise
