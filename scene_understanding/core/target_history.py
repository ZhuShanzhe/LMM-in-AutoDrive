"""Timestamp-based last-seen memory. Historical observations never become live tracks."""

from copy import deepcopy
import math


class TargetHistory:
    def __init__(self, retention_s=30., maximum_entries=8192):
        if not math.isfinite(retention_s) or retention_s<=0:
            raise ValueError('Positive finite retention required')
        self.retention_s=float(retention_s)
        self.maximum_entries=maximum_entries
        self.reset()

    def reset(self):
        self.entries={}
        self.clock=None
        self.source_clocks={}

    def _advance(self, timestamp_s):
        now=float(timestamp_s)
        if not math.isfinite(now):
            raise ValueError('Finite monotonic timestamp required')
        if self.clock is not None and now<self.clock-1e-6:
            self.reset()
        self.clock=now
        self.entries={key:value for key,value in self.entries.items()
            if now-value['last_seen_s']<=self.retention_s+1e-6}
        return now

    def update(self, source, observed_tracks, timestamp_s):
        stamp=float(timestamp_s)
        if not math.isfinite(stamp):
            raise ValueError('Finite measurement timestamp required')
        source=str(source)
        if stamp<self.source_clocks.get(source,-math.inf)-1e-6:
            return self.snapshot(source,self.clock)
        now=self._advance(max(stamp,self.clock if self.clock is not None else stamp))
        self.source_clocks[source]=stamp
        seen=set()
        for track in observed_tracks:
            identity=str(track['track_id']);key=(str(source),identity)
            if key in seen:
                raise ValueError('Duplicate observed identity in one source frame')
            seen.add(key)
            previous=self.entries.get(key)
            if previous is None and len(self.entries)>=self.maximum_entries:
                raise RuntimeError('Target history capacity exhausted; no silent early eviction')
            self.entries[key]=dict(source=str(source),track_id=identity,
                first_seen_s=previous['first_seen_s'] if previous else stamp,last_seen_s=stamp,
                currently_observed=True,
                last_observation=deepcopy(track),
                identity_evidence='SAME_UPSTREAM_TRACK_ID' if previous else 'NEW_UPSTREAM_TRACK_ID')
        for key,entry in self.entries.items():
            if key[0]==str(source) and key not in seen:
                entry['currently_observed']=False
        return self.snapshot(source,now)

    def snapshot(self, source, timestamp_s):
        now=self._advance(timestamp_s)
        records=[]
        for (stream,_),entry in self.entries.items():
            if stream!=str(source):continue
            item=deepcopy(entry);age=max(0.,now-item['last_seen_s'])
            current=item.pop('currently_observed') and age<=1e-6
            item.update(status='CURRENT' if current else 'HISTORICAL',age_s=age,
                expires_at_s=item['last_seen_s']+self.retention_s,
                usable_as_current_observation=current,
                historical_position_extrapolated=False)
            records.append(item)
        return dict(schema_version='target_history/1.0',status='AVAILABLE',source=str(source),
            timestamp_s=now,retention_s=self.retention_s,entries=records,
            cross_id_reidentification=False,authority='REFERENCE_CONTEXT_ONLY')
