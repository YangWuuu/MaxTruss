#pragma once

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <string>

using namespace std::chrono;

class Clock {
 public:
  explicit Clock(const std::string &name) { name_ = name; }

  const char *Start() {
    startTime_ = duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count();
    sprintf(str_, "[\033[1m\033[44;30m%-12s\033[0m] Start...", name_.c_str());
    return str_;
  }

  const char *Count(const char *fmt = "", ...) {
    va_list args;
    char str2[1000];
    va_start(args, fmt);
    vsprintf(str2, fmt, args);
    va_end(args);
    uint64_t end_time = duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count();
    double t = double(end_time - startTime_) / 1e6;
    sprintf(str_, "[\033[1m\033[44;31m%-12s\033[0m] %.6lfs   %s", name_.c_str(), t, str2);
    return str_;
  }

 private:
  char str_[1000]{};
  std::string name_{};
  uint64_t startTime_{};
};
