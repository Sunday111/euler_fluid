#pragma once

#include <EverydayTools/Time/MeasureTime.hpp>
#include <chrono>
#include <functional>
#include <vector>

#include "EverydayTools/Math/Math.hpp"
#include "EverydayTools/Math/Matrix.hpp"

namespace euler_fluid
{

using edt::Vec2f;
using Vec2uz = edt::Vec2<size_t>;

class EulerFluidSimulation
{
public:
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<float, std::milli>;

    class UpdateDuration
    {
    public:
        Duration integrate;
        Duration solve;
        Duration extrapolate;
        Duration advect_velocity;
        Duration advect_smoke;
        Duration total;
    };

    EulerFluidSimulation(float density, Vec2uz in_grid_size, float cell_height)
        : grid_size{in_grid_size + 2},  // sentinels
          max_coord{grid_size - 1},
          nx{grid_size.x()},
          ny{grid_size.y()},
          density_{density},
          h{cell_height},
          h1{1.f / h},
          h2{h / 2},
          sampling_delta_u_{0, h2},
          sampling_delta_v_{h2, 0},
          sampling_delta_s_{h2, h2}
    {
        num_cells_ = grid_size.x() * grid_size.y();
        s.resize(num_cells_, 0.f);
        uv_.resize(num_cells_, Vec2f{});
        new_uv_.resize(num_cells_, Vec2f{});
        m_.resize(num_cells_, 0.f);
        p.resize(num_cells_, 0.f);
        new_m_.resize(num_cells_, 0.f);

        std::ranges::fill(m_, 1.f);
    }

    UpdateDuration Simulate(float dt, size_t num_iterations, float gravity, float over_relaxation)
    {
        using Self = EulerFluidSimulation;

        UpdateDuration d{};

        d.total = edt::MeasureTime(
            [&]
            {
                d.integrate = edt::MeasureTime(std::bind_front(&Self::Integrate, this, dt, gravity));
                std::ranges::fill(p, 0.f);

                d.solve = edt::MeasureTime(
                    std::bind_front(&Self::SolveIncompressibility, this, num_iterations, dt, over_relaxation));

                d.extrapolate = edt::MeasureTime(std::bind_front(&Self::Extrapolate, this));
                d.advect_velocity = edt::MeasureTime(std::bind_front(&Self::AdvectVelocity, this, dt));
                d.advect_smoke = edt::MeasureTime(std::bind_front(&Self::AdvectSmoke, this, dt));
            });

        return d;
    }

    [[nodiscard]] float& v(size_t i) { return uv_[i].y(); }
    [[nodiscard]] const float& v(size_t i) const { return uv_[i].y(); }

    [[nodiscard]] float& u(size_t i) { return uv_[i].x(); }
    [[nodiscard]] const float& u(size_t i) const { return uv_[i].x(); }

    [[nodiscard]] float& m(size_t i) { return m_[i]; }
    [[nodiscard]] const float& m(size_t i) const { return m_[i]; }

private:
    // Step 1: modify velocity value
    void Integrate(float dt, float gravity)
    {
        for (size_t i = 1; i != nx; i++)
        {
            for (size_t j = 1; j != max_coord.y(); j++)
            {
                if (s[i * ny + j] != 0.f && s[i * ny + j - 1] != 0.f)
                {
                    v(i * ny + j) += gravity * dt;
                }
            }
        }
    }

    // Step 2: Making the fluid incompressible
    void SolveIncompressibility(size_t num_iterations, float dt, float over_relaxation)
    {
        float cp = density_ * h / dt;

        for (size_t iter = 0; iter != num_iterations; iter++)
        {
            for (size_t i = 1; i != max_coord.x(); i++)
            {
                for (size_t j = 1; j != max_coord.y(); j++)
                {
                    if (s[i * ny + j] == 0.f) continue;

                    float sx0 = s[(i - 1) * ny + j];
                    float sx1 = s[(i + 1) * ny + j];
                    float sy0 = s[i * ny + j - 1];
                    float sy1 = s[i * ny + j + 1];
                    float sum = sx0 + sx1 + sy0 + sy1;

                    if (sum == 0.f) continue;

                    float div = u((i + 1) * ny + j) - u(i * ny + j) + v(i * ny + j + 1) - v(i * ny + j);

                    float pk = -div / sum;
                    pk *= over_relaxation;
                    p[i * ny + j] += cp * pk;
                    u(i * ny + j) -= sx0 * pk;
                    u((i + 1) * ny + j) += sx1 * pk;
                    v(i * ny + j) -= sy0 * pk;
                    v(i * ny + j + 1) += sy1 * pk;
                }
            }
        }
    }

    void Extrapolate()
    {
        for (size_t i = 0; i != nx; i++)
        {
            u(i * ny + 0) = u(i * ny + 1);
            u(i * ny + ny - 1) = u(i * ny + ny - 2);
        }

        for (size_t j = 0; j != ny; j++)
        {
            v(0 * ny + j) = v(1 * ny + j);
            v((nx - 1) * ny + j) = v((nx - 2) * ny + j);
        }
    }

    [[nodiscard]] constexpr float AvgU(size_t i, size_t j) const
    {
        return (u(i * ny + j - 1) + u(i * ny + j) + u((i + 1) * ny + j - 1) + u((i + 1) * ny + j)) * 0.25f;
    }

    [[nodiscard]] constexpr float AvgV(size_t i, size_t j) const
    {
        return (v((i - 1) * ny + j) + v(i * ny + j) + v((i - 1) * ny + j + 1) + v(i * ny + j + 1)) * 0.25f;
    }

    using ValueGetterMethod = const float& (EulerFluidSimulation::*)(size_t i) const;

    template <auto getter>
    [[nodiscard]] constexpr float SampleField(Vec2f xy, const Vec2f& delta) const
    {
        xy = edt::Math::Clamp(xy, {h, h}, grid_size.Cast<float>() * h);
        auto p0 = edt::Math::ElementwiseMin(((xy - delta) * h1).Cast<size_t>(), grid_size - 1);
        auto [tx, ty] = (((xy - delta) - p0.Cast<float>() * h) * h1).Tuple();
        auto [x1, y1] = edt::Math::ElementwiseMin(p0 + 1, max_coord).Tuple();

        float sx = 1.f - tx;
        float sy = 1.f - ty;
        float a = std::invoke(getter, this, p0.x() * ny + p0.y());
        float b = std::invoke(getter, this, x1 * ny + p0.y());
        float c = std::invoke(getter, this, x1 * ny + y1);
        float d = std::invoke(getter, this, p0.x() * ny + y1);
        return sx * sy * a + tx * sy * b + tx * ty * c + sx * ty * d;
    }

    void AdvectVelocity(float dt)
    {
        new_uv_ = uv_;
        constexpr ValueGetterMethod u_getter = &EulerFluidSimulation::u;
        constexpr ValueGetterMethod v_getter = &EulerFluidSimulation::v;

        for (size_t i = 1; i != nx; i++)
        {
            float fi = static_cast<float>(i);
            for (size_t j = 1; j != ny; j++)
            {
                const auto fij = Vec2f{fi, static_cast<float>(j)};

                // u component
                if (s[i * ny + j] != 0.f && s[(i - 1) * ny + j] != 0.f && j + 1 < ny)
                {
                    Vec2f uv{u(i * ny + j), AvgV(i, j)};
                    auto xy = fij * h + sampling_delta_u_ - dt * uv;
                    new_uv_[i * ny + j].x() = SampleField<u_getter>(xy, sampling_delta_u_);
                }

                // v component
                if (s[i * ny + j] != 0.f && s[i * ny + j - 1] != 0.f && i + 1 < nx)
                {
                    Vec2f uv{AvgU(i, j), v(i * ny + j)};
                    Vec2f xy = fij * h + sampling_delta_v_ - dt * uv;
                    new_uv_[i * ny + j].y() = SampleField<v_getter>(xy, sampling_delta_v_);
                }
            }
        }

        uv_ = new_uv_;
    }

    void AdvectSmoke(float dt)
    {
        new_m_ = m_;
        constexpr ValueGetterMethod m_getter = &EulerFluidSimulation::m;

        for (size_t i = 1; i < nx - 1; i++)
        {
            for (size_t j = 1; j < ny - 1; j++)
            {
                if (s[i * ny + j] != 0.f)
                {
                    auto uv = Vec2f{u(i * ny + j) + u((i + 1) * ny + j), v(i * ny + j) + v(i * ny + j + 1)} * 0.5f;
                    auto xy = Vec2uz{i, j}.Cast<float>() * h + h2 - dt * uv;
                    new_m_[i * ny + j] = SampleField<m_getter>(xy, sampling_delta_s_);
                }
            }
        }

        m_ = new_m_;
    }

    [[nodiscard]] constexpr size_t CoordToIdx(size_t x, size_t y) const noexcept { return x * grid_size.y() + y; }

public:
    Vec2uz grid_size{100, 100};
    Vec2uz max_coord = grid_size - 1;
    size_t nx = grid_size.x();
    size_t ny = grid_size.y();
    float density_ = 1000.f;
    float h = {};
    float h1 = 1.f / h;
    float h2 = h / 2;
    Vec2f sampling_delta_u_{0, h2};
    Vec2f sampling_delta_v_{h2, 0};
    Vec2f sampling_delta_s_{h2, h2};
    size_t num_cells_{};
    std::vector<float> s{};
    std::vector<Vec2f> uv_{};
    std::vector<Vec2f> new_uv_{};
    std::vector<float> m_{};
    std::vector<float> new_m_{};
    std::vector<float> p{};
};
}  // namespace euler_fluid
