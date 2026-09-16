"""Translate Earth Engine CRS codes into forms pyproj/rasterio can parse."""

import warnings

import ee
import pyproj

# EE-only codes (spatialreference.org SR-ORG / etc.) that PROJ cannot resolve,
# mapped to an equivalent WKT string. Can extend as new codes are encountered.
#
# NOTE: these are hand-verified against EE's actual pixel grid, NOT copied from
# ``ee.Projection(code).wkt()``. For SR-ORG:6974 the EE-serialized WKT reports a
# WGS 84 ellipsoid, but EE's projection engine (and the real MODIS data) use a
# sphere of radius 6371007.181 m -- the two disagree by up to ~19 km toward the
# poles. Always confirm a new entry against EE before adding it (see the WKT
# fallback warning below).
_EE_CRS_WKT = {
    'SR-ORG:6974': (  # MODIS sinusoidal (spherical, R=6371007.181)
        'PROJCS["MODIS Sinusoidal",'
        'GEOGCS["Unknown datum based upon the custom spheroid",'
        'DATUM["Not specified (based on custom spheroid)",'
        'SPHEROID["Custom spheroid",6371007.181,0]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
        'PROJECTION["Sinusoidal"],PARAMETER["longitude_of_center",0],'
        'PARAMETER["false_easting",0],PARAMETER["false_northing",0],'
        'UNIT["Meter",1]]'
    ),
}


def to_pyproj_crs(crs):
    """Return a CRS string that pyproj/rasterio can parse.

    Known EE-only codes are translated via the static lookup table; anything
    pyproj already understands (EPSG, WKT, proj4) passes through unchanged; an
    unknown, unparseable code falls back to Earth Engine's WKT.

    This is idempotent: a resolved WKT hits the pyproj fast path on a second
    call (no extra EE round-trip), so it is safe to call more than once.

    The EE WKT fallback is a last resort: EE's serialized WKT is not guaranteed
    to match EE's actual pixel grid (e.g. SR-ORG:6974 serializes as WGS 84 but is
    really a sphere), so a warning is emitted and a verified entry should be added
    to ``_EE_CRS_WKT`` instead.
    """
    if crs in _EE_CRS_WKT:
        return _EE_CRS_WKT[crs]
    try:
        pyproj.CRS.from_user_input(crs)
        return crs
    except pyproj.exceptions.CRSError:
        warnings.warn(
            f"CRS '{crs}' is not parseable by PROJ and has no verified entry in "
            "_EE_CRS_WKT; falling back to Earth Engine's WKT, which may not match "
            "EE's pixel grid. Verify against EE and add a checked entry to "
            "_EE_CRS_WKT.",
            UserWarning,
            stacklevel=2,
        )
        return ee.Projection(crs).wkt().getInfo()
