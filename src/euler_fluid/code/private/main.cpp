#include <imgui.h>

#include <EverydayTools/Math/FloatRange.hpp>
#include <EverydayTools/Math/Math.hpp>
#include <algorithm>
#include <ass/enum_set.hpp>

#include "ass/enum/enum_as_index_magic_enum.hpp"
#include "klgl/application.hpp"
#include "klgl/error_handling.hpp"
#include "klgl/opengl/gl_api.hpp"
#include "klgl/rendering/painter2d.hpp"
#include "klgl/ui/simple_type_widget.hpp"
#include "klgl/window.hpp"
#include "magic_enum.hpp"
#include "util.hpp"

namespace colors
{
inline constexpr edt::Vec4u8 red{255, 0, 0, 255};
inline constexpr edt::Vec4u8 green{0, 255, 0, 255};
inline constexpr edt::Vec4u8 blue{0, 0, 255, 255};
inline constexpr edt::Vec4u8 white = edt::Vec4u8{} + 255;
}  // namespace colors

namespace euler_fluid
{
enum class SceneType : uint8_t
{
    WindTunnel,
    HiresTunnel,
    Tank,
    Paint,
};

enum class FieldType
{
    U,
    V,
    S
};
}  // namespace euler_fluid

template <>
struct ass::EnumIndexConverter<euler_fluid::SceneType> : ass::EnumIndexConverter_MagicEnum<euler_fluid::SceneType>
{
};

namespace euler_fluid
{

class EulerFluidSimulation
{
public:
    EulerFluidSimulation(float density, edt::Vec2<size_t> grid_size, float cell_height)
        : grid_size_{grid_size + 2},  // sentinels
          density_{density},
          h_{cell_height}
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

    template <FieldType type>
    [[nodiscard]] constexpr float SampleField(float x, float y) const
    {
        const auto [nx, ny] = grid_size_.Tuple();
        const auto [nxf, nyf] = grid_size_.Cast<float>().Tuple();
        float h = h_;
        float h1 = 1.f / h;
        float h2 = 0.5f * h;

        x = std::clamp(x, h, nxf * h);
        y = std::clamp(y, h, nyf * h);

        float dx = 0.f;
        float dy = 0.f;
        std::span<const float> f;

        if constexpr (type == FieldType::U)
        {
            f = u_;
            dy = h2;
        }

        if constexpr (type == FieldType::V)
        {
            f = v_;
            dx = h2;
        }

        if constexpr (type == FieldType::S)
        {
            f = m_;
            dx = h2;
            dy = h2;
        }

        size_t x0 = std::min(static_cast<size_t>(std::floor((x - dx) * h1)), nx - 1);
        float tx = ((x - dx) - static_cast<float>(x0) * h) * h1;
        size_t x1 = std::min(x0 + 1, nx - 1);

        size_t y0 = std::min(static_cast<size_t>(std::floor((y - dy) * h1)), ny - 1);
        float ty = ((y - dy) - static_cast<float>(y0) * h) * h1;
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

        float h = h_;
        float h2 = 0.5f * h_;

        for (size_t i = 1; i != nx; i++)
        {
            float fi = static_cast<float>(i);
            for (size_t j = 1; j != ny; j++)
            {
                // u component
                float fj = static_cast<float>(j);
                if (s_[i * ny + j] != 0.f && s_[(i - 1) * ny + j] != 0.f && j + 1 < ny)
                {
                    float x = fi * h;
                    float y = fj * h + h2;
                    float u = u_[i * ny + j];
                    float v = AvgV(i, j);
                    x = x - dt * u;
                    y = y - dt * v;
                    u = SampleField<FieldType::U>(x, y);
                    new_u_[i * ny + j] = u;
                }

                // v component
                if (s_[i * ny + j] != 0.f && s_[i * ny + j - 1] != 0.f && i + 1 < nx)
                {
                    float x = fi * h + h2;
                    float y = fj * h;
                    float u = AvgU(i, j);
                    float v = v_[i * ny + j];
                    x = x - dt * u;
                    y = y - dt * v;
                    v = SampleField<FieldType::V>(x, y);
                    new_v_[i * ny + j] = v;
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
        float h2 = 0.5f * h_;

        for (size_t i = 1; i < nx - 1; i++)
        {
            for (size_t j = 1; j < ny - 1; j++)
            {
                if (s_[i * ny + j] != 0.f)
                {
                    float u = (u_[i * ny + j] + u_[(i + 1) * ny + j]) * 0.5f;
                    float v = (v_[i * ny + j] + v_[i * ny + j + 1]) * 0.5f;
                    float x = static_cast<float>(i) * h_ + h2 - dt * u;
                    float y = static_cast<float>(j) * h_ + h2 - dt * v;

                    new_m_[i * ny + j] = SampleField<FieldType::S>(x, y);
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
    size_t num_cells_{};
    std::vector<float> s_;
    std::vector<float> u_;
    std::vector<float> v_;
    std::vector<float> m_;
    std::vector<float> p_;
    std::vector<float> new_u_;
    std::vector<float> new_v_;
    std::vector<float> new_m_;
};

class EulerFluidApp : public klgl::Application
{
    void Initialize() override
    {
        klgl::Application::Initialize();
        klgl::OpenGl::SetClearColor({});
        SetTargetFramerate(60.f);
        GetWindow().SetSize(1000, 1000);
        GetWindow().SetTitle("Painter 2d");
        painter_ = std::make_unique<klgl::Painter2d>();
        SetupScene(SceneType::HiresTunnel);
    }

    void Tick() override
    {
        const auto sim_begin = std::chrono::high_resolution_clock::now();
        fluid_->Simulate(dt_, num_iterations_, gravity_, over_relaxation_);
        const auto sim_end = std::chrono::high_resolution_clock::now();

        painter_->BeginDraw();
        painter_->DrawCircle({.center = {-0.3f, 0.3f}, .size = {.2f, .2f}, .color = colors::green});
        painter_->DrawCircle({.center = {0.3f, 0.3f}, .size = {.2f, .2f}, .color = colors::green});
        painter_->DrawCircle({.center = {0, -0.25f}, .size = {.8f, .2f}, .color = colors::green});

        auto& f = *fluid_;
        constexpr auto render_area = edt::FloatRange2Df::FromMinMax({-1, -1}, {1, 1});
        constexpr auto render_area_size = render_area.Extent();
        const auto rect_size = render_area_size / f.grid_size_.Cast<float>();
        const auto [min_p, max_p] = std::ranges::minmax_element(f.p_);

        const auto [nx, ny] = f.grid_size_.Tuple();

        for (size_t x = 0; x != nx; ++x)
        {
            for (size_t y = 0; y != ny; ++y)
            {
                edt::Vec2<size_t> coord{x, y};
                edt::Vec2f coordf = coord.Cast<float>();

                edt::Vec4u8 color{};

                if (show_pressure_)
                {
                    float p = f.p_[x * ny + y];
                    float s = f.m_[x * ny + y];
                    color = GetSciColor(p, *min_p, *max_p);
                    if (show_smoke_)
                    {
                        color.x() = static_cast<uint8_t>(std::max(0.0f, static_cast<float>(color.x()) - 255.f * s));
                        color.y() = static_cast<uint8_t>(std::max(0.0f, static_cast<float>(color.y()) - 255.f * s));
                        color.z() = static_cast<uint8_t>(std::max(0.0f, static_cast<float>(color.z()) - 255.f * s));
                    }
                }
                else if (show_smoke_)
                {
                    float s = f.m_[x * ny + y];
                    color[0] = static_cast<uint8_t>(255.f * s);
                    color[1] = static_cast<uint8_t>(255.f * s);
                    color[2] = static_cast<uint8_t>(255.f * s);
                    if (scene_type_ == SceneType::Paint)
                    {
                        color = GetSciColor(s, 0.0, 1.0);
                    }
                }
                else if (f.s_[x * ny + y] == 0.f)
                {
                    color = {};
                }

                color.w() = 255;

                painter_->DrawRect({
                    .center = (coordf + 0.5f) * rect_size + render_area.Min(),
                    .size = rect_size,
                    .color = color,
                });
            }
        }

        if (show_obstacle_)
        {
            painter_->DrawCircle({
                .center = render_area.Uniform(obstacle_),
                .size = 2 * obstacle_radius_ * render_area_size,
                .color = colors::white,
            });
        }

        painter_->EndDraw();
        if (ImGui::Begin("Settings"))
        {
            const float framerate = GetFramerate();
            klgl::SimpleTypeWidget("framerate", framerate);

            const auto ms =
                std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(sim_end - sim_begin).count();
            klgl::SimpleTypeWidget("sim ms", ms);

            if (edt::Vec2f obstacle_pos = obstacle_; klgl::SimpleTypeWidget("obstacle", obstacle_pos))
            {
                SetObstacle(obstacle_pos, true);
            }

            bool mouse_current_state = ImGui::IsMouseDown(ImGuiMouseButton_Left);
            if (ImGui::GetIO().WantCaptureMouse)
            {
                is_mouse_down_ = false;
                mouse_current_state = false;
            }

            const bool reset_obstacle = mouse_current_state && !is_mouse_down_;
            is_mouse_down_ = mouse_current_state;
            if (is_mouse_down_)
            {
                auto window_size = GetWindow().GetSize2f();
                auto im_mouse_pos = ImGui::GetMousePos();
                edt::Vec2f mouse_pos{im_mouse_pos.x, window_size.y() - im_mouse_pos.y};

                edt::Vec2f new_obstacle_position = mouse_pos / window_size;
                if (new_obstacle_position != obstacle_)
                {
                    SetObstacle(mouse_pos / window_size, reset_obstacle);
                }
            }

            ImGui::Checkbox("Pressure", &show_pressure_);
            ImGui::Checkbox("Smoke", &show_smoke_);

            for (const SceneType scene_type : ass::EnumSet<SceneType>::Full())
            {
                if (ImGui::Button(magic_enum::enum_name(scene_type).data()) && scene_type != scene_type_)
                {
                    SetupScene(scene_type);
                }
            }

            ImGui::End();
        }
    }

    void SetupScene(const SceneType scene_type)
    {
        dt_ = 1.f / 60.f;
        scene_type_ = scene_type;
        obstacle_radius_ = 0.15f;
        over_relaxation_ = 1.9f;
        num_iterations_ = 40;

        const float res = [&]
        {
            switch (scene_type)
            {
            case SceneType::Tank:
                return 50.f;
            case SceneType::HiresTunnel:
                return 200.f;
            default:
                return 100.f;
            }
        }();

        float simHeight = 1.f;
        float simWidth = 1.f;
        float domainHeight = 1.0;
        float domainWidth = domainHeight / simHeight * simWidth;
        float cell_height = domainHeight / res;

        const size_t numX = static_cast<size_t>(std::floor(domainWidth / cell_height));
        const size_t numY = static_cast<size_t>(std::floor(domainHeight / cell_height));
        fluid_ = std::make_unique<EulerFluidSimulation>(density_, edt::Vec2<size_t>(numX, numY), cell_height);

        const auto [nx, ny] = fluid_->grid_size_.Tuple();
        const auto [nxf, nyf] = fluid_->grid_size_.Cast<float>().Tuple();
        auto& f = *fluid_;
        switch (scene_type_)
        {
        case SceneType::Tank:
        {
            for (size_t i = 0; i != nx; i++)
            {
                for (size_t j = 0; j != ny; j++)
                {
                    float s = 1;                                 // fluid
                    if (i == 0 || i == nx - 1 || j == 0) s = 0;  // solid
                    f.s_[i * ny + j] = s;
                }
            }
            gravity_ = -9.81f;
            show_pressure_ = true;
            show_smoke_ = true;
            show_streamlines_ = false;
            show_velocities_ = false;
        }
        break;

        case SceneType::HiresTunnel:
        case SceneType::WindTunnel:
        {
            float inVel = 2.0;
            for (size_t i = 0; i != nx; i++)
            {
                for (size_t j = 0; j != ny; j++)
                {
                    float s = 1.0;                                 // fluid
                    if (i == 0 || j == 0 || j == ny - 1) s = 0.0;  // solid
                    f.s_[i * ny + j] = s;

                    if (i == 1)
                    {
                        f.u_[i * ny + j] = inVel;
                    }
                }
            }

            float pipeH = 0.1f * nyf;
            size_t minJ = static_cast<size_t>(std::floor((nyf - pipeH) / 2));
            size_t maxJ = static_cast<size_t>(std::floor((nyf + pipeH) / 2));

            for (size_t j = minJ; j < maxJ; j++)
            {
                f.m_[j] = 0.0f;
            }

            SetObstacle({0.4f, 0.5f}, true);

            gravity_ = 0.0f;
            show_pressure_ = false;
            show_smoke_ = true;
            show_streamlines_ = false;
            show_velocities_ = false;

            if (scene_type_ == SceneType::HiresTunnel)
            {
                dt_ = 1.f / 120.f;
                num_iterations_ = 100;
                show_pressure_ = true;
            }
        }
        break;
        case SceneType::Paint:
        {
            gravity_ = 0.0;
            over_relaxation_ = 1.0;
            show_pressure_ = false;
            show_smoke_ = true;
            show_streamlines_ = false;
            show_velocities_ = false;
            obstacle_radius_ = 0.1f;
        }
        break;
        }
    }

    void SetObstacle(edt::Vec2f position, bool reset)
    {
        edt::Vec2f v{0, 0};

        if (!reset) v = (position - obstacle_) / dt_;

        obstacle_ = position;
        float squared_radius = obstacle_radius_ * obstacle_radius_;
        auto& f = *fluid_;
        const auto [nx, ny] = fluid_->grid_size_.Tuple();

        for (size_t i = 1; i < nx - 2; i++)
        {
            for (size_t j = 1; j < ny - 2; j++)
            {
                f.s_[i * ny + j] = 1;

                const edt::Vec2f delta = (edt::Vec2<size_t>(i, j).Cast<float>() + .5f) * f.h_ - position;
                if (delta.SquaredLength() < squared_radius)
                {
                    f.s_[i * ny + j] = 0;
                    if (scene_type_ == SceneType::Paint)
                    {
                        f.m_[i * ny + j] = .5f + .5f * std::sin(GetTimeSeconds());
                    }
                    else
                    {
                        f.m_[i * ny + j] = 1;
                    }

                    f.u_[i * ny + j] = v.x();
                    f.u_[(i + 1) * ny + j] = v.x();
                    f.v_[i * ny + j] = v.y();
                    f.v_[i * ny + j + 1] = v.y();
                }
            }
        }

        show_obstacle_ = true;
    }

    std::unique_ptr<klgl::Painter2d> painter_;

    float dt_ = 0.f;
    SceneType scene_type_ = SceneType::Tank;
    float density_ = 1000.f;
    float obstacle_radius_ = 0.f;
    float over_relaxation_ = 0.f;
    float gravity_ = -9.81f;
    size_t num_iterations_ = 40;
    bool show_pressure_ = true;
    bool show_smoke_ = false;
    bool show_streamlines_ = false;
    bool show_velocities_ = false;
    bool show_obstacle_ = false;
    bool is_mouse_down_ = false;
    edt::Vec2f obstacle_{};

    std::unique_ptr<EulerFluidSimulation> fluid_;
};

}  // namespace euler_fluid

void Main()
{
    euler_fluid::EulerFluidApp app;
    app.Run();
}

int main()
{
    klgl::ErrorHandling::InvokeAndCatchAll(Main);
    return 0;
}
