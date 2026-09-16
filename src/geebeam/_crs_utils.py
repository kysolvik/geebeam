"""Translate Earth Engine CRS codes into forms pyproj/rasterio can parse."""

import ee
import pyproj

# EE-only codes (spatialreference.org SR-ORG / etc.) that PROJ cannot resolve,
# mapped to an equivalent WKT string. Can extend as new codes are encountered.
_EE_CRS_WKT = {
    'SR-ORG:6974': (  # MODIS sinusoidal
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
    """
    if crs in _EE_CRS_WKT:
        return _EE_CRS_WKT[crs]
    try:
        pyproj.CRS.from_user_input(crs)
        return crs
    except pyproj.exceptions.CRSError:
        return ee.Projection(crs).wkt().getInfo()
