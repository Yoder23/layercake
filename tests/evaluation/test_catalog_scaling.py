from layercake.evaluation.catalog_scaling import CATALOG_SIZES


def test_final_catalog_sizes_cover_required_stress_points():
    assert CATALOG_SIZES == (0, 1, 5, 10, 50, 100)
