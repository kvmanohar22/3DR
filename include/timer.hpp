#ifndef _TIMER_HPP_
#define _TIMER_HPP_

#include <chrono>
#include <map>
#include <string>
#include <iostream>
#include <iterator>

namespace dr3 {

using namespace std::chrono;

typedef high_resolution_clock hrs;

/*********************** Timer ***********************/
class Timer {
private:
   duration<double> _total_time;
   size_t _n_calls;
   hrs::time_point _start;
   hrs::time_point _end;
   duration<double> _avg_time;

public:
   Timer() {
      _total_time = duration<double>(0.0f);
      _avg_time = duration<double>(0.0f);
      _n_calls = 0;
   }

   void tic() {
      _start = hrs::now();
   }

   void toc() {
      _end = hrs::now();
      _total_time += hrs::now() - _start;
      _n_calls += 1;
      _avg_time += _total_time;
   }

   inline double get_total_time() const { return _total_time.count(); }
   inline double get_avg_time() const  { return _avg_time.count() / _n_calls; }
   inline size_t n_calls() const { return _n_calls; }

   inline void reset() {
      _total_time = duration<double>(0.0f);
      _avg_time   = duration<double>(0.0f);
      _n_calls = 0;
   }

   bool operator < (const Timer &timer) const {
      if (this->get_total_time() < timer.get_total_time())
         return true;
      return false;
   }
};

/*********************** Monitor ***********************/
class Monitor {
public:
   Monitor();
   ~Monitor();
   void add_timer(const std::string &name);
   void start_timer(const std::string &name);
   void stop_timer(const std::string &name);
   double get_time(const std::string &name) const;
   void reset(const std::string &name);

private:
   std::map<std::string, Timer> _timers;
};

} // namespace dr3

#endif
