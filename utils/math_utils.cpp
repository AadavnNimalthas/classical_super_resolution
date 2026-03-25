#include "math_utils.h"
#include <cmath>

namespace MathUtil {

    double clamp(double x, double minVal, double maxVal) {
        if (x < minVal) { return minVal; }
        if (x > maxVal) { return maxVal; }
        return x;
    }

    double sinc(double x) {
        if (x == 0.0) { return 1.0; }
        x *= M_PI;
        return sin(x) / x;
    }

    double lanczos(double x, int a) {
        if (x < -a || x > a) { return 0.0; }
        return sinc(x) * sinc(x / a);
    }

}