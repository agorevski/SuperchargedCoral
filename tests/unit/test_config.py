from supercharged_coral.common.config import platform_config_from_mapping


def test_platform_config_from_mapping_validates_cameras_and_regions():
    config = platform_config_from_mapping(
        {
            "database_path": "data/test.db",
            "cameras": [
                {
                    "id": "front-door",
                    "source_type": "synthetic",
                    "fps": 30,
                    "ignore_regions": [{"x": 1, "y": 2, "width": 3, "height": 4}],
                }
            ],
            "motion": {"algorithm": "frame_differencing", "threshold": None},
        }
    )

    assert config.cameras[0].id == "front-door"
    assert config.cameras[0].ignore_regions[0].width == 3
    assert config.motion.threshold is None
    assert "Person" in config.object_classes

