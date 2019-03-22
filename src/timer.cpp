#include "timer.hpp"

namespace dr3 {

Monitor::Monitor() {}

Monitor::~Monitor() {}

void Monitor::add_timer(const std::string &name) {
   _timers.insert(std::make_pair(name, Timer()));
}

void Monitor::start_timer(const std::string &name) {
   auto _timer = _timers.find(name);
   if (_timer == _timers.end()) {
      std::cerr << "Timer: \"" << name << "\" not registered." << std::endl;
   }
   _timer->second.tic();
}

void Monitor::stop_timer(const std::string &name) {
   auto _timer = _timers.find(name);
   if (_timer == _timers.end()) {
      std::cerr << "Timer: \"" << name << "\" not registered." << std::endl;
   }
   _timer->second.toc();
}

void Monitor::get_time(const std::string &name) {
   auto _timer = _timers.find(name);
   if (_timer == _timers.end()) {
      std::cerr << "Timer: \"" << name << "\" not registered." << std::endl;
   }
   _timer->second.get_total_time();
}

void Monitor::reset(const std::string &name) {
   auto _timer = _timers.find(name);
   if (_timer == _timers.end()) {
      std::cerr << "Timer: \"" << name << "\" not registered." << std::endl;
   }
   _timer->second.reset();
}

} // namespace dr3
