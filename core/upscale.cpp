#include "upscale.h"
#include "lanczos.h"

Image upscale_lanczos(const Image& input, int scale, int a)
{
    int new_w = input.width * scale;
    int new_h = input.height * scale;

    Image output(new_w, new_h, input.channels);

    for(int y = 0; y < new_h; y++)
    {
        for(int x = 0; x < new_w; x++)
        {
            float src_x = static_cast<float>(x) / scale;
            float src_y = static_cast<float>(y) / scale;

            for(int c = 0; c < input.channels; c++)
            {
                output.at(x, y, c) =
                    sample_lanczos(input, src_x, src_y, c, a);
            }
        }
    }

    return output;
}