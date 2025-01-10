//===-- ocn/OceanInit.cpp - Ocean Initialization ----------------*- C++ -*-===//
//
// This file contians ocnInit and associated methods which initialize Omega.
// The ocnInit process reads the config file and uses the config options to
// initialize time management and call all the individual initialization
// routines for each module in Omega.
//
//===----------------------------------------------------------------------===//

#include "AuxiliaryState.h"
#include "Config.h"
#include "DataTypes.h"
#include "Decomp.h"
#include "Field.h"
#include "Halo.h"
#include "HorzMesh.h"
#include "IO.h"
#include "IOStream.h"
#include "Logging.h"
#include "MachEnv.h"
#include "OceanDriver.h"
#include "OceanState.h"
#include "Tendencies.h"
#include "TimeMgr.h"
#include "TimeStepper.h"
#include "Tracers.h"

#include "mpi.h"

#include "roctracer/roctx.h"

namespace OMEGA {

int ocnInit(MPI_Comm Comm ///< [in] ocean MPI communicator
) {

   I4 Err = 0; // Error code

   roctxRangePush("MachEnv::init() & getDefault()");
   // Init the default machine environment based on input MPI communicator
   MachEnv::init(Comm);
   MachEnv *DefEnv = MachEnv::getDefault();
   roctxRangePop();

   // Initialize Omega logging
   roctxRangePush("initLogging()");
   initLogging(DefEnv);
   roctxRangePop();

   // Read config file into Config object
   roctxRangePush("Config() & readAll()");
   Config("Omega");
   Err = Config::readAll("omega.yml");
   roctxRangePop();
   if (Err != 0) {
      LOG_CRITICAL("ocnInit: Error reading config file");
      return Err;
   }
   Config *OmegaConfig = Config::getOmegaConfig();

   // initialize remaining Omega modules
   roctxRangePush("initOmegaModules()");
   Err = initOmegaModules(Comm);
   roctxRangePop();
   if (Err != 0) {
      LOG_CRITICAL("ocnInit: Error initializing Omega modules");
      return Err;
   }

   return Err;
} // end ocnInit

// Call init routines for remaining Omega modules
int initOmegaModules(MPI_Comm Comm) {

   // error code
   I4 Err = 0;

   // Initialize the default time stepper (phase 1) that includes the
   // calendar, model clock and start/stop times and alarms
   roctxRangePush("TimeStepper::init1()");
   Err = TimeStepper::init1();
   roctxRangePop();
   if (Err != 0) {
      LOG_CRITICAL("ocnInit: Error phase 1 initializing default time stepper");
      return Err;
   }

   TimeStepper *DefStepper = TimeStepper::getDefault();
   Clock *ModelClock       = DefStepper->getClock();

   // Initialize IOStreams - this does not yet validate the contents
   // of each file, only creates streams from Config
   roctxRangePush("IOStream::init()");
   Err = IOStream::init(ModelClock);
   roctxRangePop();
   if (Err != 0) {
      LOG_CRITICAL("ocnInit: Error initializing IOStreams");
      return Err;
   }

   roctxRangePush("IO::init()");
   Err = IO::init(Comm);
   roctxRangePop();
   if (Err != 0) {
      LOG_CRITICAL("ocnInit: Error initializing parallel IO");
      return Err;
   }

   roctxRangePush("Field::init()");
   Err = Field::init(ModelClock);
   roctxRangePop();
   if (Err != 0) {
      LOG_CRITICAL("ocnInit: Error initializing Fields");
      return Err;
   }

   roctxRangePush("Decomp::init()");
   Err = Decomp::init();
   roctxRangePop();
   if (Err != 0) {
      LOG_CRITICAL("ocnInit: Error initializing default decomposition");
      return Err;
   }

   roctxRangePush("Halo::init()");
   Err = Halo::init();
   roctxRangePop();
   if (Err != 0) {
      LOG_CRITICAL("ocnInit: Error initializing default halo");
      return Err;
   }

   roctxRangePush("HorzMesh::init()");
   Err = HorzMesh::init();
   roctxRangePop();
   if (Err != 0) {
      LOG_CRITICAL("ocnInit: Error initializing default mesh");
      return Err;
   }

   // Create the vertical dimension - this will eventually move to
   // a vertical mesh later
   roctxRangePush("Create Vert Dim");
   Config *OmegaConfig = Config::getOmegaConfig();
   Config DimConfig("Dimension");
   Err = OmegaConfig->get(DimConfig);
   if (Err != 0) {
      LOG_CRITICAL("ocnInit: Dimension group not found in Config");
      return Err;
   }
   I4 NVertLevels;
   Err = DimConfig.get("NVertLevels", NVertLevels);
   if (Err != 0) {
      LOG_CRITICAL("ocnInit: NVertLevels not found in Dimension Config");
      return Err;
   }
   auto VertDim = OMEGA::Dimension::create("NVertLevels", NVertLevels);
   roctxRangePop();

   roctxRangePush("Tracers::init()");
   Err = Tracers::init();
   roctxRangePop();
   if (Err != 0) {
      LOG_CRITICAL("ocnInit: Error initializing tracers infrastructure");
      return Err;
   }

   roctxRangePush("AuxiliaryState::init()");
   Err = AuxiliaryState::init();
   roctxRangePop();
   if (Err != 0) {
      LOG_CRITICAL("ocnInit: Error initializing default aux state");
      return Err;
   }

   roctxRangePush("Tendencies::init()");
   Err = Tendencies::init();
   roctxRangePop();
   if (Err != 0) {
      LOG_CRITICAL("ocnInit: Error initializing default tendencies");
      return Err;
   }

   roctxRangePush("TimeStepper::init2()");
   Err = TimeStepper::init2();
   roctxRangePop();
   if (Err != 0) {
      LOG_CRITICAL("ocnInit: Error phase 2 initializing default time stepper");
      return Err;
   }

   roctxRangePush("OceanState::init()");
   Err = OceanState::init();
   roctxRangePop();
   if (Err != 0) {
      LOG_CRITICAL("ocnInit: Error initializing default state");
      return Err;
   }

   // Now that all fields have been defined, validate all the streams
   // contents
   roctxRangePush("IOStream::validateAll()");
   bool StreamsValid = IOStream::validateAll();
   roctxRangePop();
   if (!StreamsValid) {
      LOG_CRITICAL("ocnInit: Error validating IO Streams");
      return Err;
   }

   // Initialize data from Restart or InitialState files
   roctxRangePush("Init Restart");
   std::string SimTimeStr          = " "; // create SimulationTime metadata
   std::shared_ptr<Field> SimField = Field::get(SimMeta);
   SimField->addMetadata("SimulationTime", SimTimeStr);

   // read from initial state if this is starting a new simulation
   Metadata ReqMeta; // no requested metadata for initial state
   Err = IOStream::read("InitialState", ModelClock, ReqMeta);
   if (Err != 0) {
      LOG_CRITICAL("Error reading the initial state file");
      return Err;
   }

   // read restart if starting from restart
   SimTimeStr                = " ";
   ReqMeta["SimulationTime"] = SimTimeStr;
   Err = IOStream::read("RestartRead", ModelClock, ReqMeta);
   if (Err != 0) {
      LOG_CRITICAL("Error reading the restart file");
      return Err;
   }

   // If reading from restart, reset the current time to the input time
   if (SimTimeStr != " ") {
      TimeInstant NewCurrentTime(SimTimeStr);
      Err = ModelClock->setCurrentTime(NewCurrentTime);
      if (Err != 0) {
         LOG_CRITICAL("Error resetting the simulation time from restart");
         return Err;
      }
   }
   roctxRangePop();

   // Update Halos and Device arrays with new state and tracer fields

   OceanState *DefState = OceanState::getDefault();
   I4 CurTimeLevel      = 0;

   roctxRangePush("DefState->exchangeHalo()");
   DefState->exchangeHalo(CurTimeLevel);
   roctxRangePop();

   DefState->copyToDevice(CurTimeLevel);

   // Now update tracers - assume using same time level index
   roctxRangePush("Tracers->exchangeHalo()");
   Err = Tracers::exchangeHalo(CurTimeLevel);
   roctxRangePop();
   if (Err != 0) {
      LOG_CRITICAL("Error updating tracer halo after restart");
      return Err;
   }
   Err = Tracers::copyToDevice(CurTimeLevel);
   if (Err != 0) {
      LOG_CRITICAL("Error updating tracer device arrays after restart");
      return Err;
   }

   return Err;

} // end initOmegaModules

} // end namespace OMEGA
//===----------------------------------------------------------------------===//
