"""Phase-aware engineering spacing audit; not a road-safety certification."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class SpacingCriteria:
    standstill_gap_m: float = 6.
    headway_s: float = 1.4
    band_min_m: float = 3.
    band_fraction: float = .25
    settle_dwell_s: float = 3.
    minimum_hold_s: float = 10.
    excursion_grace_s: float = 3.
    sampling_dt_s: float = .05
    minimum_observed_gap_m: float = 2.
    critical_ttc_s: float = 2.

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('Spacing criteria must be finite and positive')


def _runs(rows, predicate, dt):
    runs=[];start=last=None
    for row in rows:
        now=row['t_s']
        selected=predicate(row)
        if start is not None and (not selected or now-last > dt*1.5):
            runs.append(dict(start_s=start,end_s=last,duration_s=last-start+dt))
            start=last=None
        if selected:
            if start is None:start=now
            last=now
    if start is not None:runs.append(dict(start_s=start,end_s=last,duration_s=last-start+dt))
    return runs


def audit_spacing(rows, *, collision_events=0, criteria=None):
    c=criteria or SpacingCriteria();annotated=[];previous=None
    for row in rows:
        if row.get('gap_m') is None:
            raise ValueError('Spacing assessment requires a measured lead-vehicle gap')
        values=[row[k] for k in ('t_s','gap_m','speed_kmh','lead_speed_kmh')]
        if not all(math.isfinite(x) for x in values) or (previous is not None and row['t_s']<=previous):
            raise ValueError('Invalid or unordered spacing observations')
        previous=row['t_s']
        v=max(0.,row['speed_kmh']/3.6);lead=max(0.,row['lead_speed_kmh']/3.6)
        target=c.standstill_gap_m+c.headway_s*v
        width=max(c.band_min_m,c.band_fraction*target)
        gap=row['gap_m'];closing=v-lead
        ttc=gap/closing if closing>.5 else None
        zone='near' if gap<target-width else 'far' if gap>target+width else 'in_band'
        hazard=(gap<c.minimum_observed_gap_m and row['speed_kmh']>1.) or (ttc is not None and ttc<c.critical_ttc_s)
        annotated.append(dict(t_s=row['t_s'],gap_m=gap,target_gap_m=target,lower_gap_m=target-width,
            upper_gap_m=target+width,zone=zone,closing_mps=closing,ttc_s=ttc,hazard=hazard,
            speed_difference_kmh=row['speed_kmh']-row['lead_speed_kmh']))
    if not annotated:raise ValueError('No spacing observations')
    holds=_runs(annotated,lambda r:r['zone']=='in_band',c.sampling_dt_s)
    settled=[r for r in holds if r['duration_s']>=c.settle_dwell_s-1e-6]
    first=settled[0] if settled else None
    acquired_at=first['start_s']+c.settle_dwell_s-c.sampling_dt_s if first else None
    after=[r for r in annotated if acquired_at is not None and r['t_s']>acquired_at]
    departures=_runs(after,lambda r:r['zone']!='in_band',c.sampling_dt_s)
    longest_departure=max((r['duration_s'] for r in departures),default=0.)
    remaining=annotated[-1]['t_s']-acquired_at if acquired_at is not None else 0.
    hazard_rows=[r for r in annotated if r['hazard']]
    gaps=[r['gap_m'] for r in annotated]
    ttc=[r['ttc_s'] for r in annotated if r['ttc_s'] is not None]
    if collision_events or hazard_rows:status='observed_hazard'
    elif not first:
        status='insufficient_acquisition_observation' if annotated[-1]['zone']=='in_band' else 'not_acquired_within_observation'
    elif longest_departure>c.excursion_grace_s+1e-6:status='acquired_then_lost_spacing'
    elif remaining<c.minimum_hold_s:status='insufficient_post_acquisition_observation'
    else:status='meets_provisional_spacing_checks'
    return dict(status=status,criteria=asdict(c),provisional_engineering_criteria=True,
        safety_scope='Collision, minimum moving gap and closing TTC screens only; no braking-envelope certification',
        frames=len(annotated),observed_duration_s=annotated[-1]['t_s']-annotated[0]['t_s']+c.sampling_dt_s,
        first_band_entry_s=holds[0]['start_s'] if holds else None,first_confirmed_acquisition_s=acquired_at,
        post_acquisition_observation_s=remaining,longest_continuous_band_hold_s=max((r['duration_s'] for r in holds),default=0.),
        longest_post_acquisition_excursion_s=longest_departure,
        substantial_post_acquisition_excursions=sum(r['duration_s']>c.excursion_grace_s for r in departures),
        band_intervals=holds,post_acquisition_excursions=departures,
        minimum_gap_m=min(gaps),maximum_gap_m=max(gaps),initial_gap_m=gaps[0],final_gap_m=gaps[-1],
        final_target_gap_m=annotated[-1]['target_gap_m'],minimum_closing_ttc_s=min(ttc) if ttc else None,
        collision_events=collision_events,observed_hazard_frames=len(hazard_rows),
        approach_frames=sum(r['zone']=='far' for r in annotated),
        approach_faster_than_lead_frames=sum(r['zone']=='far' and r['closing_mps']>0 for r in annotated),
        phase_rows=annotated)
