//===-- ForwardBackwardStepper.cpp - forward-backward methods --*- C++ -*--===//
//
// Contains methods for the Forward-Backward time stepper
//
//===----------------------------------------------------------------------===//

#include "ForwardBackwardStepper.h"

#include "roctracer/roctx.h"

namespace OMEGA {

//------------------------------------------------------------------------------
// Constructor creates an instance of a forward-backward stepper and
// fills with some time information. Data pointers are added later.
// Mostly passes relevant info to the base constructor.
ForwardBackwardStepper::ForwardBackwardStepper(
    const std::string &InName,      ///< [in] name of time stepper
    const TimeInstant &InStartTime, ///< [in] start time for time stepping
    const TimeInstant &InStopTime,  ///< [in] stop  time for time stepping
    const TimeInterval &InTimeStep  ///< [in] time step
    )
    : TimeStepper(InName, TimeStepperType::ForwardBackward, 2, InStartTime,
                  InStopTime, InTimeStep) {}

//------------------------------------------------------------------------------
// Advance the state by one step of the forward-backward scheme
void ForwardBackwardStepper::doStep(
    OceanState *State,   // input model state
    TimeInstant &SimTime // current simulation time
) const {

   int Err = 0;

   const int CurLevel  = 0;
   const int NextLevel = 1;

   Array3DReal CurTracerArray, NextTracerArray;
   Err = Tracers::getAll(CurTracerArray, CurLevel);
   Err = Tracers::getAll(NextTracerArray, NextLevel);

   if (State == nullptr)
      LOG_CRITICAL("Invalid State");
   if (AuxState == nullptr)
      LOG_CRITICAL("Invalid AuxState");

   // R_h^{n} = RHS_h(u^{n}, h^{n}, t^{n})
   roctxRangePush("computeThicknessTendencies()");
   Tend->computeThicknessTendencies(State, AuxState, CurLevel, CurLevel,
                                    SimTime);
   roctxRangePop();

   // h^{n+1} = h^{n} + R_h^{n}
   roctxRangePush("updateThicknessByTend()");
   updateThicknessByTend(State, NextLevel, State, CurLevel, TimeStep);
   roctxRangePop();

   // R_phi^{n} = RHS_phi(u^{n}, h^{n}, phi^{n}, t^{n})
   roctxRangePush("computeTracerTendencies()");
   Tend->computeTracerTendencies(State, AuxState, CurTracerArray, CurLevel,
                                 CurLevel, SimTime);
   roctxRangePop();

   // phi^{n+1} = (phi^{n} * h^{n} + R_phi^{n}) / h^{n+1}
   roctxRangePush("updateTracersByTend()");
   updateTracersByTend(NextTracerArray, CurTracerArray, State, NextLevel, State,
                       CurLevel, TimeStep);
   roctxRangePop();

   // R_u^{n+1} = RHS_u(u^{n}, h^{n+1}, t^{n+1})
   roctxRangePush("computeVelocityTendencies()");
   Tend->computeVelocityTendencies(State, AuxState, NextLevel, CurLevel,
                                   SimTime + TimeStep);
   roctxRangePop();

   // u^{n+1} = u^{n} + R_u^{n+1}
   roctxRangePush("updateVelocityByTend()");
   updateVelocityByTend(State, NextLevel, State, CurLevel, TimeStep);
   roctxRangePop();

   // Update time levels (New -> Old) of prognostic variables with halo
   // exchanges
   roctxRangePush("State->updateTimeLevels()");
   State->updateTimeLevels();
   roctxRangePop();

   roctxRangePush("Tracers::updateTimeLevels()");
   Tracers::updateTimeLevels();
   roctxRangePop();

   // Advance the clock and update the simulation time
   roctxRangePush("StepClock->advance()");
   Err     = StepClock->advance();
   SimTime = StepClock->getCurrentTime();
   roctxRangePop();
}

} // namespace OMEGA
