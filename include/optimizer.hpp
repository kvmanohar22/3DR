#ifndef _OPTIMIZER_HPP_
#define _OPTIMIZER_HPP_

namespace 3dr {

enum class OType {
   LOCAL,
   GLOBAL,
   MOTION_ONLY
};


class Optimizer {
private:
   OType _type;


public:   
   Optimizer();

}; // class Optimizer

} // namespace 3dr 

#endif

