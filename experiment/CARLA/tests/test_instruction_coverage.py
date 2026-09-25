import pytest

from benchmark.coverage import profile_coverage,instruction_status
from benchmark.catalog import ConfigError


def test_existing_unreviewed_profile_does_not_prove_instruction_success():
    assert instruction_status('SUCCESS',profile_coverage({}))=='UNVERIFIED'


def test_partial_profile_cannot_report_instruction_success():
    coverage=profile_coverage({'coverage':{'status':'PARTIAL','missing':['gap_check']}})
    assert instruction_status('SUCCESS',coverage)=='UNVERIFIED'
    assert instruction_status('FAILURE',coverage)=='FAILURE'


def test_review_reference_required_for_complete_label():
    with pytest.raises(ConfigError):
        profile_coverage({'coverage':{'status':'COMPLETE','missing':[]}})


def test_reviewed_coverage_still_requires_criterion_success():
    coverage=profile_coverage({'coverage':{'status':'COMPLETE','missing':[],
                                         'review_reference':'review.md#task'}})
    assert instruction_status('SUCCESS',coverage)=='SUCCESS'
    assert instruction_status('TIMEOUT',coverage)=='TIMEOUT'
