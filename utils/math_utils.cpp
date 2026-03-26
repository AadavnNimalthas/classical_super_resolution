// 1.0
#include "math_utils.h"
#include <cmath>

namespace MathUtil {

    constexpr double PI = 3.14159265358979323846;

    double clamp(double x, double minVal, double maxVal)
    {
        if(x < minVal) { return minVal; }
        if(x > maxVal) { return maxVal; }
        return x;
    }

    double sinc(double x)
    {
        if(std::abs(x) < 1e-8) { return 1.0; }
        x *= PI;
        return std::sin(x) / x;
    }

    double lanczos(double x, int a)
    {
        if(x < -a || x > a) { return 0.0; }
        return sinc(x) * sinc(x / a);
    }

}