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
#include <omp.h>

#include "base.hpp"
#include "bfps_timer.hpp"

//< To add it as friend of EventManager
class ScopeEvent;

class EventManager {
protected:

    class CoreEvent {
     protected:
      //< Name of the event (from the user)
      const std::string m_name;
      //< Previous events (stack of parents)
      std::stack<CoreEvent*> m_parentStack;
      //< Current event children
      std::vector<CoreEvent*> m_children;

      //< Total execution time
      double m_totalTime;
      //< Minimum execution time
      double m_minTime;
      //< Maximum execution time
      double m_maxTime;
      //< Number of occurrence for this event
      int m_occurrence;
      //< Number of occurrence that are tasks for this event
      int m_nbTasks;
      //< Children lock
      omp_lock_t m_childrenLock;
      //< Children lock
      omp_lock_t m_updateLock;

     public:
      /** Create a core-event from the name and the current stack */
      CoreEvent(const std::string& inName,
                const std::stack<CoreEvent*>& inParentStack)
          : m_name(inName),
            m_parentStack(inParentStack),
            m_totalTime(0),
            m_minTime(std::numeric_limits<double>::max()),
            m_maxTime(std::numeric_limits<double>::min()),
            m_occurrence(0),
            m_nbTasks(0) {
        omp_init_lock(&m_childrenLock);
        omp_init_lock(&m_updateLock);
      }

      ~CoreEvent() {
        omp_destroy_lock(&m_childrenLock);
        omp_destroy_lock(&m_updateLock);
      }

      /** Add a record */
      void addRecord(const double inDuration, const bool isTask) {
  #pragma omp atomic update
        m_totalTime += inDuration;
  #pragma omp atomic update
        m_occurrence += 1;
  #pragma omp flush  // (m_minTime, m_maxTime)
        if (inDuration < m_minTime || m_maxTime < inDuration) {
          omp_set_lock(&m_updateLock);
          m_minTime = std::min(m_minTime, inDuration);
          m_maxTime = std::max(m_maxTime, inDuration);
          omp_unset_lock(&m_updateLock);
        }
        if (isTask) {
  #pragma omp atomic update
          m_nbTasks += 1;
        }
      }

      const std::stack<CoreEvent*>& getParents() const { return m_parentStack; }

      std::stack<CoreEvent*>& getParents() { return m_parentStack; }

      void addChild(CoreEvent* inChild) {
        omp_set_lock(&m_childrenLock);
        m_children.push_back(inChild);
        omp_unset_lock(&m_childrenLock);
      }

      //! Must not be called during a paralle execution
      const std::vector<CoreEvent*>& getChildren() const {
        assert(omp_in_parallel() == 0);
        return m_children;
      }

      const std::string& getName() const { return m_name; }

      double getMin() const { return m_minTime; }

      double getMax() const { return m_maxTime; }

      int getOccurrence() const { return m_occurrence; }

      double getAverage() const {
        return m_totalTime / static_cast<double>(m_occurrence);
      }

      double getDuration() const { return m_totalTime; }

      int getNbTasks() const { return m_nbTasks; }
    };

    ///////////////////////////////////////////////////////////////

    //< The main node
    std::unique_ptr<CoreEvent> m_root;
    //< Output stream to print out
    std::ostream& m_outputStream;

    //< Current stack, there are one stack of stack per thread
    std::vector<std::stack<std::stack<CoreEvent*>>> m_currentEventsStackPerThread;
    //< All recorded events (that will then be delete at the end)
    std::unordered_multimap<std::string, CoreEvent*> m_records;
    //< Lock for m_records
    omp_lock_t m_recordsLock;

    /** Find a event from its name. If such even does not exist
   * the function creates one. If an event with the same name exists
   * but with a different stack, a new one is created.
   * It pushes the returned event in the stack.
   */
    CoreEvent* getEvent(const std::string& inName,
                        const std::string& inUniqueKey) {
        const std::string completeName = inName + inUniqueKey;
        CoreEvent* foundEvent = nullptr;

        omp_set_lock(&m_recordsLock);
        // find all events with this name
        auto range = m_records.equal_range(completeName);
        for (auto iter = range.first; iter != range.second; ++iter) {
          // events are equal if same name and same parents
          if ((*iter).second->getParents() ==
              m_currentEventsStackPerThread[omp_get_thread_num()].top()) {
            foundEvent = (*iter).second;
            break;
          }
        }

        // Keep the lock to ensure that not two threads create the same event

        if (!foundEvent) {
          // create this event
          foundEvent = new CoreEvent(
              inName, m_currentEventsStackPerThread[omp_get_thread_num()].top());
          m_currentEventsStackPerThread[omp_get_thread_num()].top().top()->addChild(
              foundEvent);
          m_records.insert({completeName, foundEvent});
        }
        omp_unset_lock(&m_recordsLock);

        m_currentEventsStackPerThread[omp_get_thread_num()].top().push(foundEvent);
        return foundEvent;
    }

    CoreEvent* getEventFromContext(const std::string& inName,
                                   const std::string& inUniqueKey,
                                   const std::stack<CoreEvent*>& inParentStack) {
      m_currentEventsStackPerThread[omp_get_thread_num()].push(inParentStack);
      return getEvent(inName, inUniqueKey);
    }

    /** Pop current event */
    void popEvent(const CoreEvent* eventToRemove) {
        assert(m_currentEventsStackPerThread[omp_get_thread_num()].top().size() > 1);
        // Comparing address is cheaper
        if (m_currentEventsStackPerThread[omp_get_thread_num()].top().top() !=
            eventToRemove) {
          throw std::runtime_error(
              "You must end events (ScopeEvent/TIMEZONE) in order.\n"
              "Please make sure that you only ask to the last event to finish.");
        }
        m_currentEventsStackPerThread[omp_get_thread_num()].top().pop();
    }

    /** Pop current context */
    void popContext(const CoreEvent* eventToRemove) {
      assert(m_currentEventsStackPerThread[omp_get_thread_num()].size() > 1);
      assert(m_currentEventsStackPerThread[omp_get_thread_num()].top().size() > 1);
      // Comparing address is cheaper
      if (m_currentEventsStackPerThread[omp_get_thread_num()].top().top() !=
          eventToRemove) {
        throw std::runtime_error(
            "You must end events (ScopeEvent/TIMEZONE) in order.\n"
            "Please make sure that you only ask to the last event to finish.");
      }
      m_currentEventsStackPerThread[omp_get_thread_num()].pop();
    }

public:
    /** Create an event manager */
    EventManager(const std::string& inAppName, std::ostream& inOutputStream)
        : m_root(new CoreEvent(inAppName, std::stack<CoreEvent*>())),
          m_outputStream(inOutputStream),
          m_currentEventsStackPerThread(1) {
      m_currentEventsStackPerThread[0].emplace();
      m_currentEventsStackPerThread[0].top().push(m_root.get());
      omp_init_lock(&m_recordsLock);
    }

    ~EventManager() throw() {
        assert(m_currentEventsStackPerThread[0].size() == 1);

        assert(m_currentEventsStackPerThread[0].top().size() == 1);

        omp_destroy_lock(&m_recordsLock);

        for (auto event : m_records) {
          delete event.second;
        }
    }

    void startParallelRegion(const int inNbThreads) {
      m_currentEventsStackPerThread.resize(1);
      m_currentEventsStackPerThread.resize(inNbThreads,
                                           m_currentEventsStackPerThread[0]);
    }

    void showDistributed(const MPI_Comm inComm) const {
        int myRank, nbProcess;
        int retMpi = MPI_Comm_rank( inComm, &myRank);
        variable_used_only_in_assert(retMpi);
        assert(retMpi == MPI_SUCCESS);
        retMpi = MPI_Comm_size( inComm, &nbProcess);
        assert(retMpi == MPI_SUCCESS);

        if((&m_outputStream == &std::cout || &m_outputStream == &std::clog) && myrank != nbProcess-1){
            // Print in reverse order
            char tmp;
            retMpi = MPI_Recv(&tmp, 1, MPI_BYTE, myrank+1, 99, inComm, MPI_STATUS_IGNORE);
            assert(retMpi == MPI_SUCCESS);
        }
        m_outputStream.flush();

        std::stack<std::pair<int, const CoreEvent*>> events;

        for (int idx = static_cast<int>(m_root->getChildren().size()) - 1; idx >= 0; --idx) {
            events.push({0, m_root->getChildren()[idx]});
        }

        m_outputStream << "[TIMING-" <<  myRank<< "] Local times.\n";
        m_outputStream << "[TIMING-" <<  myRank<< "] :" << m_root->getName() << "\n";

        while (events.size()) {
            const std::pair<int, const CoreEvent*> eventToShow =
                    events.top();
            events.pop();

            m_outputStream << "[TIMING-" <<  myRank<< "] ";

            int offsetTab = eventToShow.first;
            while (offsetTab--) {
                m_outputStream << "\t";
            }
            m_outputStream << "@" << eventToShow.second->getName() << " = " << eventToShow.second->getDuration() << "s";
            if (eventToShow.second->getOccurrence() != 1) {
                m_outputStream << " (Min = " << eventToShow.second->getMin() << "s ; Max = " << eventToShow.second->getMax()
                             << "s ; Average = " << eventToShow.second->getAverage() << "s ; Occurrence = "
                             << eventToShow.second->getOccurrence() << ")";
            }

            m_outputStream << "\n";
            for (int idx =
                 static_cast<int>(eventToShow.second->getChildren().size()) - 1;
                 idx >= 0; --idx) {
                events.push(
                {eventToShow.first + 1, eventToShow.second->getChildren()[idx]});
            }
        }
        m_outputStream.flush();

        if((&m_outputStream == &std::cout || &m_outputStream == &std::clog) && myrank != 0){
            // Print in reverse order
            char tmp;
            retMpi = MPI_Send(&tmp, 1, MPI_BYTE, myrank-1, 99, inComm);
            assert(retMpi == MPI_SUCCESS);
        }
    }

    void show(const MPI_Comm inComm, const bool onlyP0 = true) const {
        int myRank, nbProcess;
        int retMpi = MPI_Comm_rank( inComm, &myRank);
        variable_used_only_in_assert(retMpi);
        assert(retMpi == MPI_SUCCESS);
        retMpi = MPI_Comm_size( inComm, &nbProcess);
        assert(retMpi == MPI_SUCCESS);

        if(onlyP0 && myRank != 0){
            return;
        }

        std::stringstream myResults;

        std::stack<std::pair<int, const CoreEvent*>> events;

        for (int idx = static_cast<int>(m_root->getChildren().size()) - 1; idx >= 0; --idx) {
            events.push({0, m_root->getChildren()[idx]});
        }

        myResults << "[TIMING-" <<  myRank<< "] Local times.\n";
        myResults << "[TIMING-" <<  myRank<< "] :" << m_root->getName() << "\n";

        while (events.size()) {
            const std::pair<int, const CoreEvent*> eventToShow =
                    events.top();
            events.pop();

            myResults << "[TIMING-" <<  myRank<< "] ";

            int offsetTab = eventToShow.first;
            while (offsetTab--) {
                myResults << "\t";
            }
            myResults << "@" << eventToShow.second->getName() << " = " << eventToShow.second->getDuration() << "s";
            if (eventToShow.second->getOccurrence() != 1) {
                myResults << " (Min = " << eventToShow.second->getMin() << "s ; Max = " << eventToShow.second->getMax()
                             << "s ; Average = " << eventToShow.second->getAverage() << "s ; Occurrence = "
                             << eventToShow.second->getOccurrence() << ")";
            }

            myResults << "\n";
            for (int idx =
                 static_cast<int>(eventToShow.second->getChildren().size()) - 1;
                 idx >= 0; --idx) {
                events.push(
                {eventToShow.first + 1, eventToShow.second->getChildren()[idx]});
            }
        }

        if(myrank != 0){
            const std::string strOutput = myResults.str();
            int sizeOutput = strOutput.length();
            retMpi = MPI_Send(&sizeOutput, 1, MPI_INT, 0, 99, inComm);
            assert(retMpi == MPI_SUCCESS);
            retMpi = MPI_Send((void*)strOutput.data(), sizeOutput, MPI_CHAR, 0, 100, inComm);
            assert(retMpi == MPI_SUCCESS);
        }
        else{
            if(onlyP0 == false){
		        std::vector<char> buffer;
		        for(int idxProc = nbProcess-1 ; idxProc > 0 ; --idxProc){
		            int sizeRecv;
		            retMpi = MPI_Recv(&sizeRecv, 1, MPI_INT, idxProc, 99, inComm, MPI_STATUS_IGNORE);
		            assert(retMpi == MPI_SUCCESS);
		            buffer.resize(sizeRecv+1);
		            retMpi = MPI_Recv(buffer.data(), sizeRecv, MPI_CHAR, idxProc, 100, inComm, MPI_STATUS_IGNORE);
		            assert(retMpi == MPI_SUCCESS);
		            buffer[sizeRecv]='\0';
		            m_outputStream << buffer.data();
		        }
			}
            m_outputStream << myResults.str();
            m_outputStream.flush();
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
        myEvents.reserve(m_records.size());

        for(const std::pair<std::string, const CoreEvent*>& event : m_records){
            myEvents.emplace_back();
            SerializedEvent& current_event = myEvents.back();

            current_event.totalTime = event.second->getDuration();
            current_event.minTime = event.second->getMin();
            current_event.maxTime = event.second->getMax();
            current_event.occurrence = event.second->getOccurrence();

            strncpy(current_event.name, event.second->getName().c_str(), 128);
            std::stringstream path;
            std::stack<CoreEvent*> parents = event.second->getParents();
            while(parents.size()){
                path << parents.top()->getName() << " << ";
                parents.pop();
            }

            strncpy(current_event.path, path.str().c_str(), 512);
        }

        // Send to process 0
        int myRank, nbProcess;
        int retMpi = MPI_Comm_rank( inComm, &myRank);
        variable_used_only_in_assert(retMpi);
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

            m_outputStream << "[MPI-TIMING] Mpi times.\n";
            for(const auto& iter : mapEvents){
                const GlobalEvent& gevent = iter.second;
                m_outputStream << "[MPI-TIMING] @" << gevent.name << "\n";
                m_outputStream << "[MPI-TIMING] Stack => " << gevent.path << "\n";
                m_outputStream << "[MPI-TIMING] \t Done by " << gevent.nbProcess << " processes\n";
                m_outputStream << "[MPI-TIMING] \t Total time for all " << gevent.totalTime
                          << "s (average per process " << gevent.totalTime/gevent.nbProcess << "s)\n";
                m_outputStream << "[MPI-TIMING] \t Min time for a process " << gevent.minTimeProcess
                          << "s Max time for a process " << gevent.maxTimeProcess << "s\n";
                m_outputStream << "[MPI-TIMING] \t The same call has been done " << gevent.occurrence
                          << " times by all process (duration min " << gevent.minTime << "s max " << gevent.maxTime << "s avg "
                          << gevent.totalTime/gevent.occurrence << "s)\n";
            }
        }
        m_outputStream.flush();
    }

    std::stack<CoreEvent*> getCurrentThreadEvent() const {
      return m_currentEventsStackPerThread[omp_get_thread_num()].top();
    }

    friend ScopeEvent;
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
class ScopeEvent {
protected:
    //< The manager to refer to
    EventManager& m_manager;
    //< The core event
    EventManager::CoreEvent* m_event;
    //< Time to get elapsed time
    bfps_timer m_timer;
    //< Is true if it has been created for task
    bool m_isTask;

public:
    ScopeEvent(const std::string& inName, EventManager& inManager,
               const std::string& inUniqueKey)
        : m_manager(inManager),
          m_event(inManager.getEvent(inName, inUniqueKey)),
          m_isTask(false) {
      m_timer.start();
    }

    ScopeEvent(const std::string& inName, EventManager& inManager,
               const std::string& inUniqueKey,
               const std::stack<EventManager::CoreEvent*>& inParentStack)
        : m_manager(inManager),
          m_event(
              inManager.getEventFromContext(inName, inUniqueKey, inParentStack)),
          m_isTask(true) {
      m_timer.start();
    }

    ~ScopeEvent() {
      m_event->addRecord(m_timer.stopAndGetElapsed(), m_isTask);
      if (m_isTask == false) {
        m_manager.popEvent(m_event);
      } else {
        m_manager.popContext(m_event);
      }
    }

    ScopeEvent(const ScopeEvent&) = delete;
    ScopeEvent& operator=(const ScopeEvent&) = delete;
    ScopeEvent(ScopeEvent&&) = delete;
    ScopeEvent& operator=(ScopeEvent&&) = delete;
};

#define ScopeEventUniqueKey_Core_To_Str_Ext(X) #X
#define ScopeEventUniqueKey_Core_To_Str(X) \
    ScopeEventUniqueKey_Core_To_Str_Ext(X)
#define ScopeEventUniqueKey __FILE__ ScopeEventUniqueKey_Core_To_Str(__LINE__)

#define ScopeEventMultiRefKey std::string("-- multiref event --")

#ifdef USE_TIMINGOUTPUT

extern EventManager global_timer_manager;

#define TIMEZONE_Core_Merge(x, y) x##y
#define TIMEZONE_Core_Pre_Merge(x, y) TIMEZONE_Core_Merge(x, y)

#define TIMEZONE(NAME)                                                      \
  ScopeEvent TIMEZONE_Core_Pre_Merge(____TIMEZONE_AUTO_ID, __LINE__)( \
      NAME, global_timer_manager, ScopeEventUniqueKey);
#define TIMEZONE_MULTI_REF(NAME)                                            \
  ScopeEvent TIMEZONE_Core_Pre_Merge(____TIMEZONE_AUTO_ID, __LINE__)( \
      NAME, global_timer_manager, ScopeEventMultiRefKey);

#define TIMEZONE_OMP_INIT_PRETASK(VARNAME)                         \
  auto VARNAME##core = global_timer_manager.getCurrentThreadEvent(); \
  auto VARNAME = &VARNAME##core;
#define TIMEZONE_OMP_TASK(NAME, VARNAME)                                    \
  ScopeEvent TIMEZONE_Core_Pre_Merge(____TIMEZONE_AUTO_ID, __LINE__)( \
      NAME, global_timer_manager, ScopeEventUniqueKey, *VARNAME);
#define TIMEZONE_OMP_PRAGMA_TASK_KEY(VARNAME) \
  shared(global_timer_manager) firstprivate(VARNAME)

#define TIMEZONE_OMP_INIT_PREPARALLEL(NBTHREADS) \
  global_timer_manager.startParallelRegion(NBTHREADS);

#else

#define TIMEZONE(NAME)
#define TIMEZONE_MULTI_REF(NAME)
#define TIMEZONE_OMP_INIT_PRETASK(VARNAME)
#define TIMEZONE_OMP_TASK(NAME, VARNAME)
#define TIMEZONE_OMP_PRAGMA_TASK_KEY(VARNAME)
#define TIMEZONE_OMP_INIT_PREPARALLEL(NBTHREADS)

#endif


#endif
