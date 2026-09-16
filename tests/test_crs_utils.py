from unittest.mock import MagicMock, patch

import pyproj

from geebeam._crs_utils import to_pyproj_crs


def test_to_pyproj_crs_ee_code_lookup():
    # SR-ORG:6974 (MODIS sinusoidal) is EE-only; PROJ cannot resolve the code
    # directly, so the lookup table must return a WKT pyproj can parse.
    result = to_pyproj_crs('SR-ORG:6974')
    parsed = pyproj.CRS.from_user_input(result)
    assert parsed.is_projected
    assert parsed.to_dict().get('proj') == 'sinu'


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
        result = to_pyproj_crs('SR-ORG:9999')
    assert result == sentinel_wkt
    mock_ctor.assert_called_once_with('SR-ORG:9999')
