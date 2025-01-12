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
#include "simulation.hpp"
#include "util.hpp"

namespace euler_fluid
{
enum class SceneType : uint8_t
{
    WindTunnel,
    HiresTunnel,
    Tank,
    Paint,
};

}  // namespace euler_fluid

template <>
struct ass::EnumIndexConverter<euler_fluid::SceneType> : ass::EnumIndexConverter_MagicEnum<euler_fluid::SceneType>
{
};

namespace colors
{
inline constexpr edt::Vec4u8 red{255, 0, 0, 255};
inline constexpr edt::Vec4u8 green{0, 255, 0, 255};
inline constexpr edt::Vec4u8 blue{0, 0, 255, 255};
inline constexpr edt::Vec4u8 white = edt::Vec4u8{} + 255;
}  // namespace colors

namespace euler_fluid
{

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
        const auto sim_stats = fluid_->Simulate(dt_, num_iterations_, gravity_, over_relaxation_);

        painter_->BeginDraw();
        painter_->DrawCircle({.center = {-0.3f, 0.3f}, .size = {.2f, .2f}, .color = colors::green});
        painter_->DrawCircle({.center = {0.3f, 0.3f}, .size = {.2f, .2f}, .color = colors::green});
        painter_->DrawCircle({.center = {0, -0.25f}, .size = {.8f, .2f}, .color = colors::green});

        auto& f = *fluid_;
        constexpr auto render_area = edt::FloatRange2Df::FromMinMax({-1, -1}, {1, 1});
        constexpr auto render_area_size = render_area.Extent();
        const auto rect_size = render_area_size / f.grid_size.Cast<float>();
        const auto [min_p, max_p] = std::ranges::minmax_element(f.p);

        const auto [nx, ny] = f.grid_size.Tuple();

        for (size_t x = 0; x != nx; ++x)
        {
            for (size_t y = 0; y != ny; ++y)
            {
                edt::Vec2<size_t> coord{x, y};
                edt::Vec2f coordf = coord.Cast<float>();

                edt::Vec4u8 color{};

                if (show_pressure_)
                {
                    float p = f.p[x * ny + y];
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
                else if (f.s[x * ny + y] == 0.f)
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

            if (ImGui::CollapsingHeader("Sim timers"))
            {
                float ms = sim_stats.integrate.count();
                klgl::SimpleTypeWidget("integrate", ms);
                ms = sim_stats.solve.count();
                klgl::SimpleTypeWidget("solve", ms);
                ms = sim_stats.extrapolate.count();
                klgl::SimpleTypeWidget("extrapolate", ms);
                ms = sim_stats.advect_smoke.count();
                klgl::SimpleTypeWidget("advect smoke", ms);
                ms = sim_stats.advect_velocity.count();
                klgl::SimpleTypeWidget("advect velocity", ms);
            }

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

        const auto [nx, ny] = fluid_->grid_size.Tuple();
        const auto [nxf, nyf] = fluid_->grid_size.Cast<float>().Tuple();
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
                    f.s[i * ny + j] = s;
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
                    f.s[i * ny + j] = s;

                    if (i == 1)
                    {
                        f.u(i * ny + j) = inVel;
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
        const auto [nx, ny] = fluid_->grid_size.Tuple();

        for (size_t i = 1; i < nx - 2; i++)
        {
            for (size_t j = 1; j < ny - 2; j++)
            {
                f.s[i * ny + j] = 1;

                const edt::Vec2f delta = (edt::Vec2<size_t>(i, j).Cast<float>() + .5f) * f.h - position;
                if (delta.SquaredLength() < squared_radius)
                {
                    f.s[i * ny + j] = 0;
                    if (scene_type_ == SceneType::Paint)
                    {
                        f.m_[i * ny + j] = .5f + .5f * std::sin(GetTimeSeconds());
                    }
                    else
                    {
                        f.m_[i * ny + j] = 1;
                    }

                    f.u(i * ny + j) = v.x();
                    f.u((i + 1) * ny + j) = v.x();
                    f.v(i * ny + j) = v.y();
                    f.v(i * ny + j + 1) = v.y();
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
