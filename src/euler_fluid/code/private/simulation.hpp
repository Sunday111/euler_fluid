#pragma once

#include <vector>

#include "EverydayTools/Math/Matrix.hpp"

namespace euler_fluid
{

using Vec2f = edt::Vec2f;
using Vec2uz = edt::Vec2<size_t>;

class EulerFluidSimulation
{
public:
    EulerFluidSimulation(float density, edt::Vec2<size_t> grid_size, float cell_height)
        : grid_size_{grid_size + 2},  // sentinels
          density_{density},
          h_{cell_height},
          h1_{1.f / h_},
          h2_{h_ / 2},
          sampling_delta_u_{0, h2_},
          sampling_delta_v_{h2_, 0},
          sampling_delta_s_{h2_, h2_}
    {
        num_cells_ = grid_size_.x() * grid_size_.y();
        s_.resize(num_cells_, 0.f);
        u_.resize(num_cells_, 0.f);
        v_.resize(num_cells_, 0.f);
        m_.resize(num_cells_, 0.f);
        p_.resize(num_cells_, 0.f);
        new_u_.resize(num_cells_, 0.f);
        new_v_.resize(num_cells_, 0.f);
        new_m_.resize(num_cells_, 0.f);

        std::ranges::fill(m_, 1.f);
    }

    void Simulate(float dt, size_t num_iterations, float gravity, float over_relaxation)
    {
        Integrate(dt, gravity);
        std::ranges::fill(p_, 0.f);
        SolveIncompressibility(num_iterations, dt, over_relaxation);

        Extrapolate();
        AdvectVelocity(dt);
        AdvectSmoke(dt);
    }

private:
    // Step 1: modify velocity value
    void Integrate(float dt, float gravity)
    {
        const auto [nx, ny] = grid_size_.Tuple();

        for (size_t i = 1; i < nx; i++)
        {
            for (size_t j = 1; j < ny - 1; j++)
            {
                if (s_[i * ny + j] != 0.f && s_[i * ny + j - 1] != 0.f)
                {
                    v_[i * ny + j] += gravity * dt;
                }
            }
        }
    }

    // Step 2: Making the fluid incompressible
    void SolveIncompressibility(size_t num_iterations, float dt, float over_relaxation)
    {
        const auto [nx, ny] = grid_size_.Tuple();
        float cp = density_ * h_ / dt;

        for (size_t iter = 0; iter != num_iterations; iter++)
        {
            for (size_t i = 1; i < nx - 1; i++)
            {
                for (size_t j = 1; j < ny - 1; j++)
                {
                    if (s_[i * ny + j] == 0.f) continue;

                    float sx0 = s_[(i - 1) * ny + j];
                    float sx1 = s_[(i + 1) * ny + j];
                    float sy0 = s_[i * ny + j - 1];
                    float sy1 = s_[i * ny + j + 1];
                    float s = sx0 + sx1 + sy0 + sy1;

                    if (s == 0.f) continue;

                    float div = u_[(i + 1) * ny + j] - u_[i * ny + j] + v_[i * ny + j + 1] - v_[i * ny + j];

                    float p = -div / s;
                    p *= over_relaxation;
                    p_[i * ny + j] += cp * p;
                    u_[i * ny + j] -= sx0 * p;
                    u_[(i + 1) * ny + j] += sx1 * p;
                    v_[i * ny + j] -= sy0 * p;
                    v_[i * ny + j + 1] += sy1 * p;
                }
            }
        }
    }

    void Extrapolate()
    {
        const auto [nx, ny] = grid_size_.Tuple();

        for (size_t i = 0; i < nx; i++)
        {
            u_[i * ny + 0] = u_[i * ny + 1];
            u_[i * ny + ny - 1] = u_[i * ny + ny - 2];
        }

        for (size_t j = 0; j < ny; j++)
        {
            v_[0 * ny + j] = v_[1 * ny + j];
            v_[(nx - 1) * ny + j] = v_[(nx - 2) * ny + j];
        }
    }

    [[nodiscard]] constexpr float AvgU(size_t i, size_t j) const
    {
        const size_t ny = grid_size_.y();
        return (u_[i * ny + j - 1] + u_[i * ny + j] + u_[(i + 1) * ny + j - 1] + u_[(i + 1) * ny + j]) * 0.25f;
    }

    [[nodiscard]] constexpr float AvgV(size_t i, size_t j) const
    {
        const size_t ny = grid_size_.y();
        return (v_[(i - 1) * ny + j] + v_[i * ny + j] + v_[(i - 1) * ny + j + 1] + v_[i * ny + j + 1]) * 0.25f;
    }

    [[nodiscard]] constexpr float SampleField(const Vec2f& xy, const std::vector<float>& f, const Vec2f& delta) const
    {
        const auto [nx, ny] = grid_size_.Tuple();
        const auto [nxf, nyf] = grid_size_.Cast<float>().Tuple();
        float h1 = 1.f / h_;

        auto [x, y] = xy.Tuple();
        auto [dx, dy] = delta.Tuple();

        x = std::clamp(x, h_, nxf * h_);
        y = std::clamp(y, h_, nyf * h_);

        size_t x0 = std::min(static_cast<size_t>(std::floor((x - dx) * h1)), nx - 1);
        float tx = ((x - dx) - static_cast<float>(x0) * h_) * h1;
        size_t x1 = std::min(x0 + 1, nx - 1);

        size_t y0 = std::min(static_cast<size_t>(std::floor((y - dy) * h1)), ny - 1);
        float ty = ((y - dy) - static_cast<float>(y0) * h_) * h1;
        size_t y1 = std::min(y0 + 1, ny - 1);

        float sx = 1.f - tx;
        float sy = 1.f - ty;
        size_t a = x0 * ny + y0;
        size_t b = x1 * ny + y0;
        size_t c = x1 * ny + y1;
        size_t d = x0 * ny + y1;
        return sx * sy * f[a] + tx * sy * f[b] + tx * ty * f[c] + sx * ty * f[d];
    }

    void AdvectVelocity(float dt)
    {
        new_u_ = u_;
        new_v_ = v_;

        const auto [nx, ny] = grid_size_.Tuple();

        for (size_t i = 1; i != nx; i++)
        {
            float fi = static_cast<float>(i);
            for (size_t j = 1; j != ny; j++)
            {
                const auto fij = edt::Vec2f{fi, static_cast<float>(j)};

                // u component
                if (s_[i * ny + j] != 0.f && s_[(i - 1) * ny + j] != 0.f && j + 1 < ny)
                {
                    Vec2f uv{u_[i * ny + j], AvgV(i, j)};
                    auto xy = fij * h_ + sampling_delta_u_ - dt * uv;
                    new_u_[i * ny + j] = SampleField(xy, u_, sampling_delta_u_);
                }

                // v component
                if (s_[i * ny + j] != 0.f && s_[i * ny + j - 1] != 0.f && i + 1 < nx)
                {
                    Vec2f uv{AvgU(i, j), v_[i * ny + j]};
                    Vec2f xy = fij * h_ + sampling_delta_v_ - dt * uv;
                    new_v_[i * ny + j] = SampleField(xy, v_, sampling_delta_v_);
                }
            }
        }

        u_ = new_u_;
        v_ = new_v_;
    }

    void AdvectSmoke(float dt)
    {
        new_m_ = m_;

        const auto [nx, ny] = grid_size_.Tuple();

        for (size_t i = 1; i < nx - 1; i++)
        {
            for (size_t j = 1; j < ny - 1; j++)
            {
                if (s_[i * ny + j] != 0.f)
                {
                    float u = (u_[i * ny + j] + u_[(i + 1) * ny + j]) * 0.5f;
                    float v = (v_[i * ny + j] + v_[i * ny + j + 1]) * 0.5f;
                    float x = static_cast<float>(i) * h_ + h2_ - dt * u;
                    float y = static_cast<float>(j) * h_ + h2_ - dt * v;

                    new_m_[i * ny + j] = SampleField({x, y}, m_, sampling_delta_s_);
                }
            }
        }

        m_ = new_m_;
    }

    [[nodiscard]] constexpr size_t CoordToIdx(size_t x, size_t y) const noexcept { return x * grid_size_.y() + y; }

public:
    edt::Vec2<size_t> grid_size_{100, 100};
    float density_ = 1000.f;
    float h_ = {};
    float h1_ = 1.f / h_;
    float h2_ = h_ / 2;
    edt::Vec2f sampling_delta_u_{0, h2_};
    edt::Vec2f sampling_delta_v_{h2_, 0};
    edt::Vec2f sampling_delta_s_{h2_, h2_};
    size_t num_cells_{};
    std::vector<float> s_{};
    std::vector<float> u_{};
    std::vector<float> v_{};
    std::vector<float> m_{};
    std::vector<float> p_{};
    std::vector<float> new_u_{};
    std::vector<float> new_v_{};
    std::vector<float> new_m_{};
};
}  // namespace euler_fluid
