#include "lanczos.h"
#include <cmath>

float sinc(float x)
{
    if(x == 0.0f) return 1.0f;
    x *= M_PI;
    return sinf(x) / x;
}

float lanczos(float x, int a)
{
    x = std::abs(x);

    if(x < a)
    {
        return sinc(x) * sinc(x / a);
    }
    return 0.0f;
}

float sample_lanczos(const Image& img, float x, float y, int c, int a)
{
    int x_int = static_cast<int>(x);
    int y_int = static_cast<int>(y);

    float result = 0.0f;
    float total_weight = 0.0f;

    for(int j = -a + 1; j <= a; j++)
    {
        for(int i = -a + 1; i <= a; i++)
        {
            int px = x_int + i;
            int py = y_int + j;

            if(px < 0 || px >= img.width || py < 0 || py >= img.height)
                continue;

            float weight =
                lanczos(x - px, a) *
                lanczos(y - py, a);

            result += img.at(px, py, c) * weight;
            total_weight += weight;
        }
    }

    if(total_weight > 0.0f)
        result /= total_weight;

    return result;
}