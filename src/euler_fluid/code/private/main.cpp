#include <imgui.h>

#include <EverydayTools/Math/Math.hpp>

#include "klgl/application.hpp"
#include "klgl/error_handling.hpp"
#include "klgl/opengl/gl_api.hpp"
#include "klgl/rendering/painter2d.hpp"
#include "klgl/window.hpp"

namespace colors
{
inline constexpr edt::Vec4u8 red{255, 0, 0, 255};
inline constexpr edt::Vec4u8 green{0, 255, 0, 255};
inline constexpr edt::Vec4u8 blue{0, 0, 255, 255};
}  // namespace colors

class Painter2dApp : public klgl::Application
{
    void Initialize() override
    {
        klgl::Application::Initialize();
        klgl::OpenGl::SetClearColor({});
        GetWindow().SetSize(1000, 1000);
        GetWindow().SetTitle("Painter 2d");
        painter_ = std::make_unique<klgl::Painter2d>();
    }

    void Tick() override
    {
        painter_->BeginDraw();
        painter_->DrawCircle({.center = {-0.3f, 0.3f}, .size = {.2f, .2f}, .color = colors::green});
        painter_->DrawCircle({.center = {0.3f, 0.3f}, .size = {.2f, .2f}, .color = colors::green});
        painter_->DrawCircle({.center = {0, -0.25f}, .size = {.8f, .2f}, .color = colors::green});

        painter_->EndDraw();
    }

    std::unique_ptr<klgl::Painter2d> painter_;
};

void Main()
{
    Painter2dApp app;
    app.Run();
}

int main()
{
    klgl::ErrorHandling::InvokeAndCatchAll(Main);
    return 0;
}
