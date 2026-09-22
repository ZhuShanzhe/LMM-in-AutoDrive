import importlib.util
from pathlib import Path


def test_directional_counts_and_contiguous_empty_duration():
    path = Path(__file__).parents[1] / 'tools/summarize_route_traffic.py'
    spec = importlib.util.spec_from_file_location('coverage_summary', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [dict(progress_m=i * 500, front_count=n, front_same_direction_count=s)
            for i, (n, s) in enumerate([(4, 3), (2, 0), (2, 0), (5, 4), (0, 0)])]
    result = module.summarize(rows, fps=2)
    assert result['counts']['front_count']['longest_empty_s'] == .5
    assert result['counts']['front_same_direction_count']['longest_empty_s'] == 1
    assert result['counts']['front_same_direction_count']['at_least_three_fraction'] == .4
    assert len(result['segments']) == 3
