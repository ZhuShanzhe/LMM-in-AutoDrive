"""Keep criterion success separate from reviewed instruction coverage."""
import copy

from .catalog import ConfigError


def profile_coverage(spec):
    value=spec.get('coverage',dict(status='NOT_REVIEWED',missing=['semantic_coverage_review']))
    if not isinstance(value,dict) or value.get('status') not in {'NOT_REVIEWED','PARTIAL','COMPLETE'}:
        raise ConfigError('invalid instruction coverage status')
    missing=value.get('missing')
    if not isinstance(missing,list) or any(not isinstance(m,str) or not m for m in missing):
        raise ConfigError('coverage missing must contain nonempty requirement names')
    if value['status']=='COMPLETE':
        if missing or not isinstance(value.get('review_reference'),str) or not value['review_reference'].strip():
            raise ConfigError('complete coverage requires a review reference and no missing requirements')
    elif not missing:
        raise ConfigError('unreviewed/partial coverage must name unresolved requirements')
    return copy.deepcopy(value)


def instruction_status(criterion_status,coverage):
    if criterion_status=='SUCCESS' and coverage['status']!='COMPLETE':
        return 'UNVERIFIED'
    return criterion_status
