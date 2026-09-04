"""
axisforge/fixtures/capabilities/__init__.py

Capability metadata only. The capability CLASSES live with the stage they
belong to -- ConstructionCapabilities in
fixtures/construction/construction_capabilities.py, StudyCapabilities in
fixtures/studies/study_capabilities.py.

Deliberately empty of imports. catalogue.py is imported BY
construction_capabilities.py, so re-exporting those classes here made this
package and that module import each other. The cycle stayed hidden only
while every caller happened to enter through this __init__ first; entering
from construction_capabilities instead raises ImportError on a partially
initialized module.

catalogue is still reachable as a submodule, no re-export needed:
    from axisforge.fixtures.capabilities import catalogue
"""