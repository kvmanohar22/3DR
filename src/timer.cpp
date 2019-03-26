#include "timer.hpp"

namespace dr3 {

Monitor::Monitor() {}

Monitor::~Monitor() {}

void Monitor::add_timer(const std::string &name) {
   _timers.insert(std::make_pair(name, Timer()));
}

void Monitor::tic(const std::string &name) {
   auto _timer = _timers.find(name);
   if (_timer == _timers.end()) {
      std::cerr << "Timer: \"" << name << "\" not registered." << std::endl;
   }
   _timer->second.tic();
}

void Monitor::toc(const std::string &name) {
   auto _timer = _timers.find(name);
   if (_timer == _timers.end()) {
      std::cerr << "Timer: \"" << name << "\" not registered." << std::endl;
   }
   _timer->second.toc();
}

double Monitor::get_ct(const std::string &name) {
   auto _timer = _timers.find(name);
   if (_timer == _timers.end()) {
      std::cerr << "Timer: \"" << name << "\" not registered." << std::endl;
   }
   return _timer->second.get_ct();
}

double Monitor::get_at(const std::string &name) {
   auto _timer = _timers.find(name);
   if (_timer == _timers.end()) {
      std::cerr << "Timer: \"" << name << "\" not registered." << std::endl;
   }
   _timer->second.reset();
   return _timer->second.get_at();
}

void Monitor::reset(const std::string &name) {
   auto _timer = _timers.find(name);
   if (_timer == _timers.end()) {
      std::cerr << "Timer: \"" << name << "\" not registered." << std::endl;
   }
   _timer->second.reset();
}

} // namespace dr3
