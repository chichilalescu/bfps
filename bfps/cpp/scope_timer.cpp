

#include "scope_timer.hpp"


#ifdef USE_TIMINGOUTPUT
scope_timer_manager global_timer_manager("BFPS", std::cout);
#endif
