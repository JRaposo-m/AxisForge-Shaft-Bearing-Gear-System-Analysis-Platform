"""axisforge/fixtures/capabilities/catalogue_data/__init__.py

Per-domain capability leaf dicts, split out of catalogue.py so that file
doesn't keep growing indefinitely as more studies (gear_iso6336, fatigue,
lubrication, failure_risk, ...) get added. catalogue.py imports each
domain's dict from here and assembles them back into the single
CATALOGUE structure -- nothing outside catalogue.py should import from
this package directly, so its own lookup/verify/introspection functions
stay the one entry point everything else already calls.
"""