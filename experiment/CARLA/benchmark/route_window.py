"""Finite route windows for the isolated Traffic Manager test baseline."""
import bisect
import math


class RouteWindow:
    def __init__(self, route, horizon_m=250):
        self.route=route
        self.distances=[point['distance_m'] for point in route]
        if (len(route)<2 or not math.isfinite(horizon_m) or horizon_m<=0
                or any(not math.isfinite(value) for value in self.distances)
                or any(b<=a for a,b in zip(self.distances,self.distances[1:]))):
            raise ValueError('route window requires increasing route distances and positive horizon')
        self.horizon=horizon_m

    def points(self, progress_m):
        if not math.isfinite(progress_m):
            raise ValueError('route progress must be finite')
        first=bisect.bisect_right(self.distances,progress_m+1)
        end=bisect.bisect_right(self.distances,progress_m+self.horizon)
        return self.route[first:max(first+1,end)]

    def at_end(self, progress_m, tolerance_m=3):
        return progress_m>=self.distances[-1]-tolerance_m
