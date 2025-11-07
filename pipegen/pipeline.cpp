#include <Halide.h>

#include <nlohmann/json.hpp>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "astvisitor.h"  // existing AST capture path (no internal headers)
#include "pipeline.h"

// NOTE: We intentionally avoid including Halide Internal headers here to keep
// compatibility with packaged Halide installs (e.g., Homebrew), which don't
// ship internal pass headers.

// Helper: simple trim (right)
static inline std::string rtrim_copy(std::string s) {
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t' ||
                        s.back() == '\r' || s.back() == '\n'))
    s.pop_back();
  return s;
}

// Helper: count leading spaces (indentation)
static inline int leading_spaces(const std::string &s) {
  int i = 0;
  while (i < (int)s.size() && s[i] == ' ') i++;
  return i;
}

// Parse Halide's print_loop_nest() text into a nested JSON.
// Root is an array. Nodes (examples):
// - { "type":"Produce"|"Consume"|"Store", "func":"f" , "body":[ ... ] }
// - { "type":"For", "for_type": "...", "var":"x", "header": "full header line",
// "body":[ ... ] }
// - { "type":"Provide", "func":"f" }  (leaf)
// - fallback: { "type":"Line", "text":"..." }
static nlohmann::json schedule_json_from_text(const std::string &txt) {
  using nlohmann::json;

  json root = json::array();

  // Stack of (indent, container-array to append children into)
  std::vector<std::pair<int, json *>> stack;
  stack.emplace_back(-1, &root);

  // Regexes for key lines
  const std::regex re_block_directive(
      R"(^\s*(produce|consume|store)\s+([A-Za-z0-9_$.]+)\s*:)");
  const std::regex re_for_header(
      R"(^\s*([A-Za-z]+)\s+([A-Za-z0-9_$.]+).*\:\s*$)");
  const std::regex re_provide(R"(^\s*([A-Za-z0-9_$.]+)\s*\()");

  std::istringstream iss(txt);
  std::string line;
  while (std::getline(iss, line)) {
    if (line.empty()) continue;
    std::string raw = rtrim_copy(line);
    if (raw.empty()) continue;

    int indent = leading_spaces(raw);

    // Pop to appropriate parent
    while (!stack.empty() && indent <= stack.back().first) {
      stack.pop_back();
    }
    if (stack.empty()) {
      // Shouldn't happen, but guard: reset to root
      stack.emplace_back(-1, &root);
    }

    json *parent = stack.back().second;

    std::smatch m;
    if (std::regex_search(raw, m, re_block_directive)) {
      // produce|consume|store
      std::string kind = m[1];
      std::string func = m[2];
      json node;
      if (kind == "produce")
        node["type"] = "Produce";
      else if (kind == "consume")
        node["type"] = "Consume";
      else
        node["type"] = "Store";
      node["func"] = func;
      node["body"] = json::array();
      parent->push_back(node);
      // Push this block's body
      json &inserted = parent->back();
      stack.emplace_back(indent, &inserted["body"]);
      continue;
    }

    if (std::regex_search(raw, m, re_for_header)) {
      // For header line: first token is loop kind, second is var (already
      // simplified by Halide)
      std::string for_type = m[1];
      std::string var = m[2];
      json node;
      node["type"] = "For";
      node["for_type"] = for_type;
      node["var"] = var;
      node["header"] = raw;  // keep full header string ("Serial x in [..]
                             // DeviceAPI:") for reference
      node["body"] = json::array();
      parent->push_back(node);
      json &inserted = parent->back();
      stack.emplace_back(indent, &inserted["body"]);
      continue;
    }

    if (std::regex_search(raw, m, re_provide)) {
      // Provide line: "f(...)=..."
      std::string func = m[1];
      json leaf;
      leaf["type"] = "Provide";
      leaf["func"] = func;
      parent->push_back(leaf);
      continue;
    }

    // Fallback: keep the line
    json other;
    other["type"] = "Line";
    other["text"] = raw;
    parent->push_back(other);
  }

  // Clean up stack (not strictly required)
  return root;
}

// Capture Halide's loop-nest printout into a std::string.
// We rely on the public Func::print_loop_nest() behavior which writes to
// std::cout. NEW: take by value (non-const)
static std::string capture_loop_nest_text(Halide::Func f) {
  std::ostringstream oss;
  auto *old = std::cout.rdbuf(oss.rdbuf());
  f.print_loop_nest();
  std::cout.rdbuf(old);
  return oss.str();
}

Pipeline::Pipeline(Halide::Func output) {
  this->output = output;
  // DFS the output function to find all functions in the DAG.
  std::vector<Halide::Func> stack;
  std::vector<Halide::Func> visited;
  stack.push_back(output);
  while (!stack.empty()) {
    Halide::Func f = stack.back();
    stack.pop_back();
    // Check if the function is already in the list.
    bool found = false;
    for (const auto &func : visited) {
      if (func.name() == f.name()) {
        found = true;
        break;
      }
    }
    if (found) {
      continue;
    }
    visited.push_back(f);
    // Get the dependencies of the function.
    auto dependencies = Halide::Internal::find_direct_calls(f.function());
    parents[f.name()] = {};
    for (const auto &dep : dependencies) {
      Halide::Func depFunc(dep.second);
      stack.push_back(depFunc);
      parents[f.name()].push_back(depFunc);
      if (children.find(dep.first) == children.end()) {
        children[dep.first] = {};
      }
      children[dep.first].push_back(f);
    }
  }
  // Topologically sort the functions.
  topologicalSort();
  halidePipeline = Halide::Pipeline(output);
}

Pipeline::Pipeline(const Pipeline &other) {
  // Deep copy the output function.
  Halide::Func copiedOutput("output");
  std::map<Halide::Internal::FunctionPtr, Halide::Internal::FunctionPtr> copies;
  other.output.function().deep_copy(copiedOutput.function().get_contents(),
                                    copies);
  *this = Pipeline(copiedOutput);
}

Pipeline::~Pipeline() {}

nlohmann::json Pipeline::serializeDAG() {
  nlohmann::json j = nlohmann::json::array();
  for (auto &func : this->funcs) {
    nlohmann::json funcJson;
    funcJson["name"] = func.name();
    funcJson["parents"] = nlohmann::json::array();
    for (auto &parent : this->parents[func.name()]) {
      funcJson["parents"].push_back(parent.name());
    }
    j.push_back(funcJson);
  }
  return j;
}

nlohmann::json Pipeline::serializeAST() {
  // Return an array of functions with their name and the AST we collected
  // during generation.
  nlohmann::json root = nlohmann::json::array();
  for (auto &func : this->funcs) {
    nlohmann::json f;
    f["name"] = func.name();
    auto it = func_asts.find(func.name());
    if (it != func_asts.end()) {
      f["ast"] = it->second;
    } else {
      f["ast"] = nullptr;
    }
    root.push_back(f);
  }
  return root;
}

nlohmann::json Pipeline::serializeSchedule() {
  // Use public API only: print the loop nest and parse to nested JSON.
  const std::string ln = capture_loop_nest_text(this->output);
  return schedule_json_from_text(ln);
}

void Pipeline::topologicalSort() {
  std::unordered_map<std::string, int> inDegree;
  for (const auto &entry : children) {
    inDegree[entry.first] = entry.second.size();
  }
  std::vector<Halide::Func> orphans = {this->output};
  funcs.clear();

  while (!orphans.empty()) {
    Halide::Func f = orphans.back();
    orphans.pop_back();
    funcs.push_back(f);
    for (const auto &parent : parents[f.name()]) {
      inDegree[parent.name()]--;
      if (inDegree[parent.name()] == 0) {
        orphans.push_back(parent);
      }
    }
  }
}
