# test_package_init.py
"""Testes de fem_solvers/__init__.py -- confirma que o ponto único de
import eager carrega sem erro e que __all__ bate certo com o que está
realmente acessível no módulo (a razão de ser desta reescrita: o
ficheiro antigo tinha nomes/caminhos que já não existiam e só
rebentavam quando alguém tentasse mesmo usá-los)."""
import axisforge.solvers.machine_elements.shaft.fem_solvers as fem_solvers


class TestPackageInit:
    def test_all_names_in___all___are_importable(self):
        missing = [name for name in fem_solvers.__all__ if not hasattr(fem_solvers, name)]
        assert missing == [], f"__all__ lista nomes não expostos: {missing}"

    def test_no_stale_names_from_old_layout(self):
        """A versão antiga expunha RigidBearingFEMSolver e apontava
        para módulos (.rigid_bearing, .sub_models) que já não existem
        -- confirma que esses nomes desapareceram, não apenas que os
        novos existem."""
        assert not hasattr(fem_solvers, "RigidBearingFEMSolver")

    def test_solvers_are_the_real_classes(self):
        from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.rigid_support import (
            RigidSupportFEMSolver as RealRigidSupportFEMSolver,
        )
        from axisforge.solvers.machine_elements.shaft.fem_solvers.submodel_solver.lagrange_multipliers import (
            SubmodelSolver as RealSubmodelSolver,
        )
        assert fem_solvers.RigidSupportFEMSolver is RealRigidSupportFEMSolver
        assert fem_solvers.SubmodelSolver is RealSubmodelSolver
