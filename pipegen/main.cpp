#include <Halide.h>
#include <spdlog/spdlog.h>

#include <vector>

int main(int argc, char *argv[]) {
  Halide::Var x("x"), y("y");
  Halide::Func f("f"), g("g"), h("h");

  f(x, y) = x + y;
  g(x, y) = f(x, y) * 2;
  h(x, y) = g(x, y);
  h(x, y) += f(x, y);

  Halide::Pipeline p(h);
  auto targets = std::vector<Halide::Target>{Halide::get_host_target()};
  p.compile_to_multitarget_static_library("pipelines/example/pipeline", {},
                                          targets);
  p.compile_to_lowered_stmt("pipelines/example/lowered.html", {}, Halide::HTML);

  return 0;
}
