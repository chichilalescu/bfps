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

#ifndef SCOPE_TIMER_HPP
#define SCOPE_TIMER_HPP

#include <memory>
#include <iostream>
#include <vector>
#include <stack>
#include <string>
#include <limits>
#include <cassert>
#include <sstream>
#include <unordered_map>
#include <mpi.h>
#include <cstring>
#include <stdexcept>

#include "base.hpp"
#include "bfps_timer.hpp"

//< To add it as friend of scope_timer_manager
class scope_timer;

class scope_timer_manager {
protected:

    class CoreEvent {
    protected:
        //< Name of the event (from the user)
        const std::string name;
        //< Previous events (stack of parents)
        std::stack<std::shared_ptr<CoreEvent>> parentStack;
        //< Current event children
        std::vector<std::shared_ptr<CoreEvent>> children;

        //< Total execution time
        double totalTime;
        //< Minimum execution time
        double minTime;
        //< Maximum execution time
        double maxTime;
        //< Number of occurrence for this event
        int occurrence;

    public:
        /** Create a core-event from the name and the current stack */
        CoreEvent(const std::string& inName,
                  const std::stack<std::shared_ptr<CoreEvent>>& inParentStack)
            : name(inName),
              parentStack(inParentStack),
              totalTime(0),
              minTime(std::numeric_limits<double>::max()),
              maxTime(std::numeric_limits<double>::min()),
              occurrence(0) {}

        /** Add a record */
        void addRecord(const double inDuration) {
            totalTime += inDuration;
            occurrence += 1;
            minTime = std::min(minTime, inDuration);
            maxTime = std::max(maxTime, inDuration);
        }

        const std::stack<std::shared_ptr<CoreEvent>>& getParents() const {
            return parentStack;
        }

        void addChild(std::shared_ptr<CoreEvent>& inChild) {
            children.push_back(inChild);
        }

        const std::vector<std::shared_ptr<CoreEvent>>& getChildren() const {
            return children;
        }

        const std::string& getName() const { return name; }

        double getMin() const { return minTime; }

        double getMax() const { return maxTime; }

        int getOccurrence() const { return occurrence; }

        double getAverage() const {
            return totalTime / static_cast<double>(occurrence);
        }

        double getDuration() const { return totalTime; }
    };

    ///////////////////////////////////////////////////////////////

    //< First event (root of all stacks)
    std::shared_ptr<CoreEvent> root;
    //< Output stream to print out
    std::ostream& outputStream;

    //< Current stack
    std::stack<std::shared_ptr<CoreEvent>> currentEventsStack;
    //< All recorded events
    std::unordered_multimap<std::string, std::shared_ptr<CoreEvent>> records;

    /** Find a event from its name. If such even does not exist
   * the function creates one. If an event with the same name exists
   * but with a different stack, a new one is created.
   * It pushes the returned event in the stack.
   */
    CoreEvent& getEvent(const std::string& inName,
                        const std::string& inUniqueKey) {
        const std::string completeName = inName + inUniqueKey;
        std::shared_ptr<CoreEvent> foundEvent;

        auto range = records.equal_range(completeName);
        for (auto iter = range.first; iter != range.second; ++iter) {
            if ((*iter).second->getParents() == currentEventsStack) {
                foundEvent = (*iter).second;
                break;
            }
        }

        if (!foundEvent) {
            foundEvent.reset(new CoreEvent(inName, currentEventsStack));
            currentEventsStack.top()->addChild(foundEvent);
            records.insert({completeName, foundEvent});
        }

        currentEventsStack.push(foundEvent);
        return (*foundEvent);
    }

    /** Pop current event */
    void popEvent(const CoreEvent& eventToRemove) {
        // Poped to many events, root event cannot be poped
        assert(currentEventsStack.size() > 1);
        // Comparing address is cheaper
        if (currentEventsStack.top().get() != &eventToRemove) {
            throw std::runtime_error(
                        "You must end events (scope_timer/TIMEZONE) in order.\n"
                        "Please make sure that you only ask to the last event to finish.");
        }
        currentEventsStack.pop();
    }

public:
    /** Create an event manager */
    scope_timer_manager(const std::string& inAppName, std::ostream& inOutputStream)
        : root(
              new CoreEvent(inAppName, std::stack<std::shared_ptr<CoreEvent>>())),
          outputStream(inOutputStream) {
        currentEventsStack.push(root);
    }

    ~scope_timer_manager() throw() {
        // Oups, the event-stack is corrupted, should be 1
        assert(currentEventsStack.size() == 1);
    }

    void show(const MPI_Comm inComm) const {
        int myRank, nbProcess;
        int retMpi = MPI_Comm_rank( inComm, &myRank);
        assert(retMpi == MPI_SUCCESS);
        retMpi = MPI_Comm_size( inComm, &nbProcess);
        assert(retMpi == MPI_SUCCESS);

        if((&outputStream == &std::cout || &outputStream == &std::clog) && myrank != nbProcess-1){
            // Print in reverse order
            char tmp;
            MPI_Recv(&tmp, 1, MPI_BYTE, myrank+1, 99, inComm, MPI_STATUS_IGNORE);
        }

        std::stack<std::pair<int, const std::shared_ptr<CoreEvent>>> events;

        for (int idx = static_cast<int>(root->getChildren().size()) - 1; idx >= 0; --idx) {
            events.push({0, root->getChildren()[idx]});
        }

        outputStream << "[TIMING-" <<  myRank<< "] Local times.\n";
        outputStream << "[TIMING-" <<  myRank<< "] :" << root->getName() << "\n";

        while (events.size()) {
            const std::pair<int, const std::shared_ptr<CoreEvent>> eventToShow =
                    events.top();
            events.pop();

            outputStream << "[TIMING-" <<  myRank<< "] ";

            int offsetTab = eventToShow.first;
            while (offsetTab--) {
                outputStream << "\t";
            }
            outputStream << "@" << eventToShow.second->getName() << " = " << eventToShow.second->getDuration() << "s";
            if (eventToShow.second->getOccurrence() != 1) {
                outputStream << " (Min = " << eventToShow.second->getMin() << "s ; Max = " << eventToShow.second->getMax()
                             << "s ; Average = " << eventToShow.second->getAverage() << "s ; Occurrence = "
                             << eventToShow.second->getOccurrence() << ")";
            }

            outputStream << "\n";
            for (int idx =
                 static_cast<int>(eventToShow.second->getChildren().size()) - 1;
                 idx >= 0; --idx) {
                events.push(
                {eventToShow.first + 1, eventToShow.second->getChildren()[idx]});
            }
        }
        outputStream.flush();

        if((&outputStream == &std::cout || &outputStream == &std::clog) && myrank != 0){
            // Print in reverse order
            char tmp;
            MPI_Send(&tmp, 1, MPI_BYTE, myrank-1, 99, inComm);
        }
    }

    void showMpi(const MPI_Comm inComm) const {
        struct SerializedEvent {
            char path[512];
            char name[128];
            double totalTime;
            double minTime;
            double maxTime;
            int occurrence;
        };

        // Convert my events into sendable object

        std::vector<SerializedEvent> myEvents;
        myEvents.reserve(records.size());

        for(const std::pair<std::string, std::shared_ptr<CoreEvent>>& event : records){
            myEvents.emplace_back();
            SerializedEvent& current_event = myEvents.back();

            current_event.totalTime = event.second->getDuration();
            current_event.minTime = event.second->getMin();
            current_event.maxTime = event.second->getMax();
            current_event.occurrence = event.second->getOccurrence();

            strncpy(current_event.name, event.second->getName().c_str(), 128);
            std::stringstream path;
            std::stack<std::shared_ptr<CoreEvent>> parents = event.second->getParents();
            while(parents.size()){
                path << parents.top()->getName() << " << ";
                parents.pop();
            }

            strncpy(current_event.path, path.str().c_str(), 512);
        }

        // Send to process 0
        int myRank, nbProcess;
        int retMpi = MPI_Comm_rank( inComm, &myRank);
        assert(retMpi == MPI_SUCCESS);
        retMpi = MPI_Comm_size( inComm, &nbProcess);
        assert(retMpi == MPI_SUCCESS);
        std::unique_ptr<int[]> nbEventsPerProc;
        if(myRank == 0){
            nbEventsPerProc.reset(new int[nbProcess]);
        }
        const int myNbEvents = myEvents.size();
        retMpi = MPI_Gather(const_cast<int*>(&myNbEvents), 1, MPI_INT,
                       nbEventsPerProc.get(), 1, MPI_INT,
                       0, inComm);
        assert(retMpi == MPI_SUCCESS);
        // Process 0 merge and print results
        std::unique_ptr<int[]> dipls;
        std::unique_ptr<SerializedEvent[]> allEvents;
        std::unique_ptr<int[]> nbEventsPerProcByte;
        std::unique_ptr<int[]> diplsByte;
        if(myRank == 0){
            dipls.reset(new int[nbProcess+1]);
            diplsByte.reset(new int[nbProcess+1]);
            nbEventsPerProcByte.reset(new int[nbProcess]);
            dipls[0] = 0;
            diplsByte[0] = 0;
            for(int idx = 1 ; idx <= nbProcess ; ++idx){
                dipls[idx] = dipls[idx-1] + nbEventsPerProc[idx-1];
                diplsByte[idx] = dipls[idx] * sizeof(SerializedEvent);
                nbEventsPerProcByte[idx-1] = nbEventsPerProc[idx-1] * sizeof(SerializedEvent);
            }
            allEvents.reset(new SerializedEvent[dipls[nbProcess]]);
        }

        retMpi = MPI_Gatherv(myEvents.data(), myNbEvents * sizeof(SerializedEvent), MPI_BYTE,
                    allEvents.get(), nbEventsPerProcByte.get(), diplsByte.get(),
                    MPI_BYTE, 0, inComm);
        assert(retMpi == MPI_SUCCESS);

        if(myRank == 0){
            struct GlobalEvent {
                char path[512];
                char name[128];
                double totalTime;
                double minTime;
                double maxTime;
                int occurrence;
                int nbProcess;
                double minTimeProcess;
                double maxTimeProcess;
            };

            std::unordered_map<std::string, GlobalEvent> mapEvents;
            for(int idxEvent = 0 ; idxEvent < dipls[nbProcess] ; ++idxEvent){
                const std::string key = std::string(allEvents[idxEvent].path) + std::string(allEvents[idxEvent].name);
                if(mapEvents.find(key) == mapEvents.end()){
                    GlobalEvent& newEvent = mapEvents[key];
                    strncpy(newEvent.path, allEvents[idxEvent].path, 512);
                    strncpy(newEvent.name, allEvents[idxEvent].name, 128);
                    newEvent.totalTime = allEvents[idxEvent].totalTime;
                    newEvent.minTime = allEvents[idxEvent].minTime;
                    newEvent.maxTime = allEvents[idxEvent].maxTime;
                    newEvent.occurrence = allEvents[idxEvent].totalTime;
                    newEvent.nbProcess = 1;
                    newEvent.minTimeProcess = allEvents[idxEvent].totalTime;
                    newEvent.maxTimeProcess = allEvents[idxEvent].totalTime;
                }
                else{
                    GlobalEvent& newEvent = mapEvents[key];
                    assert(strcmp(newEvent.path, allEvents[idxEvent].path) == 0);
                    assert(strcmp(newEvent.name, allEvents[idxEvent].name) == 0);
                    newEvent.totalTime += allEvents[idxEvent].totalTime;
                    newEvent.minTime = std::min(newEvent.minTime, allEvents[idxEvent].minTime);
                    newEvent.maxTime = std::max(newEvent.maxTime, allEvents[idxEvent].maxTime);
                    newEvent.occurrence += allEvents[idxEvent].occurrence;
                    newEvent.nbProcess += 1;
                    newEvent.minTimeProcess = std::min(newEvent.minTimeProcess,
                                                       allEvents[idxEvent].totalTime);
                    newEvent.maxTimeProcess = std::max(newEvent.maxTimeProcess,
                                                       allEvents[idxEvent].totalTime);
                }
            }

            outputStream << "[MPI-TIMING] Mpi times.\n";
            for(const auto& iter : mapEvents){
                const GlobalEvent& gevent = iter.second;
                outputStream << "[MPI-TIMING] @" << gevent.name << "\n";
                outputStream << "[MPI-TIMING] Stack => " << gevent.path << "\n";
                outputStream << "[MPI-TIMING] \t Done by " << gevent.nbProcess << " processes\n";
                outputStream << "[MPI-TIMING] \t Total time for all " << gevent.totalTime
                          << "s (average per process " << gevent.totalTime/gevent.nbProcess << "s)\n";
                outputStream << "[MPI-TIMING] \t Min time for a process " << gevent.minTimeProcess
                          << "s Max time for a process " << gevent.maxTimeProcess << "s\n";
                outputStream << "[MPI-TIMING] \t The same call has been done " << gevent.occurrence
                          << " times by all process (duration min " << gevent.minTime << "s max " << gevent.maxTime << "s avg "
                          << gevent.totalTime/gevent.occurrence << "s)\n";
            }
        }
        outputStream.flush();
    }

    friend scope_timer;
};

///////////////////////////////////////////////////////////////

/** A scope event should be used
 * to record the duration of a part of the code
 * (section, scope, etc.).
 * The timer is stoped automatically when the object is destroyed
 * or when "finish" is explicitely called.
 * The object cannot be copied/moved to ensure coherency in the
 * events hierarchy.
 */
class scope_timer {
protected:
    //< The manager to refer to
    scope_timer_manager& manager;
    //< The core event
    scope_timer_manager::CoreEvent& event;
    //< Time to get elapsed time
    bfps_timer timer;

public:
    scope_timer(const std::string& inName, scope_timer_manager& inManager,
                const std::string& inUniqueKey)
        : manager(inManager), event(inManager.getEvent(inName, inUniqueKey)) {
        timer.start();
    }

    ~scope_timer() {
        event.addRecord(timer.stopAndGetElapsed());
        manager.popEvent(event);
    }

    scope_timer(const scope_timer&) = delete;
    scope_timer& operator=(const scope_timer&) = delete;
    scope_timer(scope_timer&&) = delete;
    scope_timer& operator=(scope_timer&&) = delete;
};

#define ScopeEventUniqueKey_Core_To_Str_Ext(X) #X
#define ScopeEventUniqueKey_Core_To_Str(X) \
    ScopeEventUniqueKey_Core_To_Str_Ext(X)
#define ScopeEventUniqueKey __FILE__ ScopeEventUniqueKey_Core_To_Str(__LINE__)

#define ScopeEventMultiRefKey std::string("-- multiref event --")

#ifdef USE_TIMINGOUTPUT

extern scope_timer_manager global_timer_manager;

#define TIMEZONE_Core_Merge(x, y) x##y
#define TIMEZONE_Core_Pre_Merge(x, y) TIMEZONE_Core_Merge(x, y)

#define TIMEZONE(NAME)                                                      \
    scope_timer TIMEZONE_Core_Pre_Merge(____TIMEZONE_AUTO_ID, __LINE__)( \
    NAME, global_timer_manager, ScopeEventUniqueKey);
#define TIMEZONE_MULTI_REF(NAME)                                            \
    scope_timer TIMEZONE_Core_Pre_Merge(____TIMEZONE_AUTO_ID, __LINE__)( \
    NAME, global_timer_manager, ScopeEventMultiRefKey);

#else

#define TIMEZONE(NAME)
#define TIMEZONE_MULTI_REF(NAME)

#endif


#endif
