"""Resolve explicit model sensor requirements without inferring them from names."""


def resolve_sensor_contract(config, available_cameras, enable_lidar):
    required = config.get('sensor_requirements')
    source = 'runner_arguments'
    if required is not None:
        if not isinstance(required, dict) or set(required) != {'cameras', 'lidar'}:
            raise ValueError('sensor_requirements must contain cameras and lidar')
        available_cameras = required['cameras']
        enable_lidar = required['lidar']
        source = 'model_config.sensor_requirements'
    if (not isinstance(available_cameras, (list, tuple)) or not available_cameras
            or any(not isinstance(view, str) or view not in ('front', 'left', 'right', 'rear')
                   for view in available_cameras)
            or len(set(available_cameras)) != len(available_cameras)):
        raise ValueError('camera views must be unique supported names')
    if not isinstance(enable_lidar, bool):
        raise ValueError('lidar requirement must be boolean')
    return dict(cameras=list(available_cameras), lidar=enable_lidar, source=source)
