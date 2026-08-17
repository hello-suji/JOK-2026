import numpy as np
import pytest
from numpy.testing import assert_allclose

from astropy.coordinates import SkyCoord
from astropy.wcs.wcsapi.high_level_wcs_wrapper import HighLevelWCSWrapper
from astropy.wcs.wcsapi.low_level_api import BaseLowLevelWCS


class CustomLowLevelWCS(BaseLowLevelWCS):
    @property
    def pixel_n_dim(self):
        return 2

    @property
    def world_n_dim(self):
        return 2

    @property
    def world_axis_physical_types(self):
        return ["pos.eq.ra", "pos.eq.dec"]

    @property
    def world_axis_units(self):
        return ["deg", "deg"]

    def pixel_to_world_values(self, *pixel_arrays):
        return [np.asarray(pix) * 2 for pix in pixel_arrays]

    def world_to_pixel_values(self, *world_arrays):
        return [np.asarray(world) / 2 for world in world_arrays]

    @property
    def world_axis_object_components(self):
        return [
            ("test", 0, "spherical.lon.degree"),
            ("test", 1, "spherical.lat.degree"),
        ]

    @property
    def world_axis_object_classes(self):
        return {"test": (SkyCoord, (), {"unit": "deg"})}


def test_wrapper():
    wcs = CustomLowLevelWCS()

    wrapper = HighLevelWCSWrapper(wcs)

    coord = wrapper.pixel_to_world(1, 2)

    assert isinstance(coord, SkyCoord)
    assert coord.isscalar

    x, y = wrapper.world_to_pixel(coord)

    assert_allclose(x, 1)
    assert_allclose(y, 2)

    assert wrapper.low_level_wcs is wcs
    assert wrapper.pixel_n_dim == 2
    assert wrapper.world_n_dim == 2
    assert wrapper.world_axis_physical_types == ["pos.eq.ra", "pos.eq.dec"]
    assert wrapper.world_axis_units == ["deg", "deg"]
    assert wrapper.array_shape is None
    assert wrapper.pixel_bounds is None
    assert np.all(wrapper.axis_correlation_matrix)


def test_wrapper_invalid():
    class InvalidCustomLowLevelWCS(CustomLowLevelWCS):
        @property
        def world_axis_object_classes(self):
            return {}

    wcs = InvalidCustomLowLevelWCS()

    wrapper = HighLevelWCSWrapper(wcs)

    with pytest.raises(KeyError):

    with pytest.raises(KeyError):
        wrapper.pixel_to_world(1, 2)

def test_world_to_pixel_inconsistent_behavior_repro():
    import astropy.wcs
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    nx = 100
    ny = 25
    nz = 2
    wcs_header = {
        'WCSAXES': 3,
        'CRPIX1': (nx + 1)/2,
        'CRPIX2': (ny + 1)/2,
        'CRPIX3': 1.0,
        'PC1_1': 0.,
        'PC1_2': 1.,
        'PC1_3': 0.0001,
        'PC2_1': -1.,
        'PC2_2': 0.,
        'PC2_3': 0.,
        'PC3_1': 0.,
        'PC3_2': 0.,
        'PC3_3': 1.,
        'CDELT1': -0.002,
        'CDELT2': 0.002,
        'CDELT3': 1.,
        'CUNIT1': 'deg',
        'CUNIT2': 'deg',
        'CUNIT3': 'm',
        'CTYPE1': 'RA---TAN',
        'CTYPE2': 'DEC--TAN',
        'CTYPE3': 'WAVE',
        'CRVAL1': 10.,
        'CRVAL2': 5.,
        'CRVAL3': 1e11,
    }

    wcs = astropy.wcs.WCS(wcs_header)
    coord = SkyCoord(ra=10*u.deg, dec=5*u.deg, frame='fk5')

    # Slice the WCS to get a 2D WCS
    sliced_wcs = wcs.slice((slice(None), slice(None), 0))

    # Perform world_to_pixel on the sliced WCS
    x, y = sliced_wcs.world_to_pixel(coord.ra, coord.dec)

    # Expected result from the issue description
    assert_allclose(x, 49.5)
    assert_allclose(y, 12.)
