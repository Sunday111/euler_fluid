#pragma once

#include "EverydayTools/Math/Matrix.hpp"

[[nodiscard]] constexpr edt::Vec4u8 GetSciColor(float val, float minVal, float maxVal)
{
    val = std::min(std::max(val, minVal), maxVal - 0.0001f);
    float d = maxVal - minVal;
    val = d == 0.0f ? 0.5f : (val - minVal) / d;
    float m = 0.25;
    float num = std::floor(val / m);
    float s = (val - num * m) / m;
    float r = 0, g = 0., b = 0.;

    switch (static_cast<size_t>(num))
    {
    case 0:
        r = 0.0;
        g = s;
        b = 1.0;
        break;
    case 1:
        r = 0.0;
        g = 1.0;
        b = 1.0f - s;
        break;
    case 2:
        r = s;
        g = 1.0;
        b = 0.0;
        break;
    case 3:
        r = 1.0;
        g = 1.0f - s;
        b = 0.0;
        break;
    }

    return (edt::Vec4f{r, g, b, 1.f} * 255.f).Cast<uint8_t>();
}
