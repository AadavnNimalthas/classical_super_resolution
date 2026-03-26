// 1.0
#include "lanczos.h"
#include <cmath>

constexpr float PI = 3.14159265358979323846f;

float sinc(float x)
{
    if(x == 0.0f) return 1.0f;
    x *= PI;
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
    int x_int = static_cast<int>(std::floor(x));
    int y_int = static_cast<int>(std::floor(y));

    float result = 0.0f;
    float total_weight = 0.0f;

    for(int j = -a + 1; j <= a; j++)
    {
        for(int i = -a + 1; i <= a; i++)
        {
            int px = x_int + i;
            int py = y_int + j;

            if(px < 0) px = 0;
            if(py < 0) py = 0;
            if(px >= img.width) px = img.width - 1;
            if(py >= img.height) py = img.height - 1;

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