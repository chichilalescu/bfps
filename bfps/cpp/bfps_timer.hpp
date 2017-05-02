/**********************************************************************
*                                                                     *
*  Copyright 2015 Max Planck Institute                                *
*                 for Dynamics and Self-Organization                  *
*                                                                     *
*  This file is part of bfps.                                         *
*                                                                     *
*  bfps is free software: you can redistribute it and/or modify       *
*  it under the terms of the GNU General Public License as published  *
*  by the Free Software Foundation, either version 3 of the License,  *
*  or (at your option) any later version.                             *
*                                                                     *
*  bfps is distributed in the hope that it will be useful,            *
*  but WITHOUT ANY WARRANTY; without even the implied warranty of     *
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the      *
*  GNU General Public License for more details.                       *
*                                                                     *
*  You should have received a copy of the GNU General Public License  *
*  along with bfps.  If not, see <http://www.gnu.org/licenses/>       *
*                                                                     *
* Contact: Cristian.Lalescu@ds.mpg.de                                 *
*                                                                     *
**********************************************************************/
#ifndef BFPS_TIMER_HPP
#define BFPS_TIMER_HPP

#include <chrono>

/**
  * @file
 *
 * Each section to measure should be embraced by start/stop.
 * The measured time is given by "getElapsed".
 * The total time measured by a timer is given by "getCumulated".
 * Example :
 * @code bfps_timer tm; // Implicit start
 * @code ...
 * @code tm.stop(); // stop the timer
 * @code tm.getElapsed(); // return the duration in s [A]
 * @code tm.start(); // restart the timer
 * @code ...
 * @code tm.stopAndGetElapsed(); // stop the timer and return the duraction in s
 * [B]
 * @code tm.getCumulated(); // Equal [A] + [B]
 */
class bfps_timer {
    using double_second_time = std::chrono::duration<double, std::ratio<1, 1>>;

    std::chrono::high_resolution_clock::time_point
    m_start;  ///< m_start time (start)
    std::chrono::high_resolution_clock::time_point m_end;  ///< stop time (stop)
    std::chrono::nanoseconds m_cumulate;  ///< the m_cumulate time

public:
    /// Constructor
    bfps_timer() : m_cumulate(std::chrono::nanoseconds::zero()) { start(); }

    /// Copy constructor
    bfps_timer(const bfps_timer& other) = delete;
    /// Copies an other timer
    bfps_timer& operator=(const bfps_timer& other) = delete;
    /// Move constructor
    bfps_timer(bfps_timer&& other) = delete;
    /// Copies an other timer
    bfps_timer& operator=(bfps_timer&& other) = delete;

    /** Rest all the values, and apply start */
    void reset() {
        m_start = std::chrono::high_resolution_clock::time_point();
        m_end = std::chrono::high_resolution_clock::time_point();
        m_cumulate = std::chrono::nanoseconds::zero();
        start();
    }

    /** Start the timer */
    void start() {
        m_start = std::chrono::high_resolution_clock::now();
    }

    /** Stop the current timer */
    void stop() {
        m_end = std::chrono::high_resolution_clock::now();
        m_cumulate += std::chrono::duration_cast<std::chrono::nanoseconds>(m_end - m_start);
    }

    /** Return the elapsed time between start and stop (in second) */
    double getElapsed() const {
        return std::chrono::duration_cast<double_second_time>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(m_end - m_start)).count();
    }

    /** Return the total counted time */
    double getCumulated() const {
        return std::chrono::duration_cast<double_second_time>(m_cumulate).count();
    }

    /** End the current counter (stop) and return the elapsed time */
    double stopAndGetElapsed() {
        stop();
        return getElapsed();
    }
};

#endif
