#include <math.h>
#include <stddef.h>

int project_route(const double *xyz, const double *road, const double *lengths,
                  const double *squared, const double *offsets,
                  size_t n, size_t segments,
                  double *lateral, double *progress, double *height) {
    if (segments == 0 || segments > 200000 || n > 50000000) return -1;
    for (size_t i = 0; i < n; ++i) {
        double best = INFINITY, p = 0.0, h = 0.0;
        for (size_t j = 0; j < segments; ++j) {
            const double *a = road + 3*j, *b = a + 3;
            double vx = b[0]-a[0], vy = b[1]-a[1];
            double dx = xyz[3*i]-a[0], dy = xyz[3*i+1]-a[1];
            double f = (dx*vx + dy*vy)/squared[j];
            if (f < 0.0) f = 0.0;
            if (f > 1.0) f = 1.0;
            double rx = dx-f*vx, ry = dy-f*vy;
            double distance = sqrt(rx*rx + ry*ry);
            if (distance < best) {
                best = distance;
                p = offsets[j]+f*lengths[j];
                h = xyz[3*i+2]-(a[2]+f*(b[2]-a[2]));
            }
        }
        lateral[i] = best; progress[i] = p; height[i] = h;
    }
    return 0;
}
