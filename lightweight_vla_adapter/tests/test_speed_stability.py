from lightweight_vla_adapter.scripts.audit_speed_stability import assess_speed


def test_exactly_95_percent_in_band_passes():
    assert assess_speed([27.]*95+[26.9]*5,30.,eligible=True)['status']=='pass'


def test_stable_wrong_speed_does_not_pass():
    assert assess_speed([0.]*100,30.,eligible=True)['status']=='fail'


def test_changing_reference_phase_not_graded():
    assert assess_speed([0.]*100,30.,eligible=False)['status']=='not_applicable'
