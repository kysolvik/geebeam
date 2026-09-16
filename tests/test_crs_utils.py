from unittest.mock import MagicMock, patch

import pyproj
import pytest

from geebeam._crs_utils import _EE_CRS_WKT, to_pyproj_crs


def test_to_pyproj_crs_ee_code_lookup():
    # SR-ORG:6974 (MODIS sinusoidal) is EE-only; PROJ cannot resolve the code
    # directly, so the lookup table must return a WKT pyproj can parse.
    result = to_pyproj_crs('SR-ORG:6974')
    parsed = pyproj.CRS.from_user_input(result)
    assert parsed.is_projected
    assert parsed.to_dict().get('proj') == 'sinu'


def test_sr_org_6974_matches_ee_sphere_grid():
    # EE's SR-ORG:6974 pixel grid (and the real MODIS data) is a SPHERE of
    # radius 6371007.181 m, verified against EE's projection engine and an
    # actual MODIS export. Its serialized ee.Projection(...).wkt() reports a
    # WGS 84 ellipsoid instead, which is wrong by up to ~19 km toward the poles,
    # so the table must stay on the sphere. Pin the sphere here so it can't
    # silently regress to the ellipsoid.
    crs = pyproj.CRS.from_user_input(_EE_CRS_WKT['SR-ORG:6974'])
    assert crs.ellipsoid.semi_major_metre == pytest.approx(6371007.181, abs=1e-3)
    x, y = pyproj.Transformer.from_crs('EPSG:4326', crs, always_xy=True).transform(0, 60)
    assert y == pytest.approx(6671702.9, abs=1.0)  # sphere; WGS 84 would be ~6,654,073


def test_to_pyproj_crs_standard_codes_pass_through():
    assert to_pyproj_crs('EPSG:4326') == 'EPSG:4326'
    assert to_pyproj_crs('EPSG:32610') == 'EPSG:32610'


def test_to_pyproj_crs_fallback_to_ee_wkt():
    # An unknown, unparseable code is not in the table and cannot be parsed by
    # PROJ, so it must fall back to Earth Engine's WKT.
    sentinel_wkt = 'PROJCS["from-ee",...]'
    mock_proj = MagicMock()
    mock_proj.wkt.return_value.getInfo.return_value = sentinel_wkt
    with patch('geebeam._crs_utils.ee.Projection', return_value=mock_proj) as mock_ctor:
        with pytest.warns(UserWarning, match='may not match'):
            result = to_pyproj_crs('SR-ORG:9999')
    assert result == sentinel_wkt
    mock_ctor.assert_called_once_with('SR-ORG:9999')
