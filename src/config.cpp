#include <config.hpp>

namespace sim {

Config *Config::_instance = NULL;

Config::Config() {}

Config* Config::get_instance() {
    if (_instance == NULL)
        _instance = get_instance(); // Generate only one instance of the class
    return _instance;
}

} // namespace sim
