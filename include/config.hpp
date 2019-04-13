#ifndef _CONFIG_HPP_
#define _CONFIG_HPP_

#include <string>
#include <stdint.h>
#include <stdio.h>

namespace sim {

using std::string;

class Config {
public:
    static Config* get_instance();

private:
    Config();
    Config(Config const&);
    Config& operator = (Config const&);
    static Config *_instance;
};

} // namespace sim

#endif
