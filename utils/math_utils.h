#ifndef MATHUTIL_H
#define MATHUTIL_H

namespace MathUtil {

    double clamp(double x, double minVal, double maxVal);

    double sinc(double x);

    double lanczos(double x, int a);

}

#endif