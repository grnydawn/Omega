//===-- Test driver for OMEGA Tracers -----------------------------*- C++
//-*-===/
//
/// \file
/// \brief Test driver for OMEGA tracers class
///
/// This driver tests that the OMEGA tracers class member variables are read in
/// correctly from a sample shperical mesh file. Also tests that the time level
/// update works as expected.
//
//===-----------------------------------------------------------------------===/

#include "Tracers.h"

#include "DataTypes.h"
#include "Decomp.h"
#include "Dimension.h"
#include "Field.h"
#include "Halo.h"
#include "HorzMesh.h"
#include "IO.h"
#include "Logging.h"
#include "MachEnv.h"
#include "OmegaKokkos.h"
#include "TimeStepper.h"
#include "mpi.h"

#include <iostream>

using namespace OMEGA;

//------------------------------------------------------------------------------
// The initialization routine for Tracers testing. It calls various
// init routines, including the creation of the default decomposition.

const Real RefReal = 3.0;

int initTracersTest() {

   int Err = 0;

   // Initialize the Machine Environment class - this also creates
   // the default MachEnv. Then retrieve the default environment and
   // some needed data members.
   MachEnv::init(MPI_COMM_WORLD);
   MachEnv *DefEnv  = MachEnv::getDefault();
   MPI_Comm DefComm = DefEnv->getComm();

   initLogging(DefEnv);

   // Open config file
   Config("Omega");
   Err = Config::readAll("omega.yml");
   if (Err != 0) {
      LOG_ERROR("Tracers: Error reading config file");
      return Err;
   }

   // Initialize the IO system
   Err = IO::init(DefComm);
   if (Err != 0) {
      LOG_ERROR("Tracers: error initializing parallel IO");
      return Err;
   }

   // Create the default decomposition (initializes the decomposition)
   Err = Decomp::init();
   if (Err != 0) {
      LOG_ERROR("Tracers: error initializing default decomposition");
      return Err;
   }

   // Initialize the default halo
   Err = Halo::init();
   if (Err != 0) {
      LOG_ERROR("Tracers: error initializing default halo");
      return Err;
   }

   // Initialize the default mesh
   Err = HorzMesh::init();
   if (Err != 0) {
      LOG_ERROR("Tracers: error initializing default mesh");
      return Err;
   }

   // Initialize the default time stepper
   Err = TimeStepper::init();
   if (Err != 0) {
      LOG_ERROR("Tracers: error initializing default time stepper");
      return Err;
   }
   return 0;
}

//------------------------------------------------------------------------------
// The test driver for Tracers -> This tests the time level update of tracers
// variables and verifies the tracers is read in correctly.
//
int main(int argc, char *argv[]) {

   int RetVal = 0;
   int Ret;

   // Initialize the global MPI environment
   MPI_Init(&argc, &argv);
   Kokkos::initialize();
   {

      // Call initialization routine to create the default decomposition
      int Err = initTracersTest();
      if (Err != 0)
         LOG_ERROR("Tracers: Error initializing");
      int count = 0;

      // Get MPI vars if needed
      MachEnv *DefEnv = MachEnv::getDefault();
      MPI_Comm Comm   = DefEnv->getComm();
      I4 MyTask       = DefEnv->getMyTask();
      I4 NumTasks     = DefEnv->getNumTasks();
      bool IsMaster   = DefEnv->isMasterTask();

      HorzMesh *DefHorzMesh = HorzMesh::getDefault();
      Decomp *DefDecomp     = Decomp::getDefault();
      Halo *DefHalo         = Halo::getDefault();

      // 2.1 Requirement: Tracer definition and metadata
      // 2.2 Requirement: Tracer identification
      // 2.3 Requirement: Tracer groups
      // 2.4 Requirement: Tracer selection
      //
      //       - Check if OceanTracers correctly include the tracer definitins
      //       in an external "TracerDefs.inc" file
      //       - Check if OceanTracers correctly generates metadata(Field) of
      //       each tracer definitions
      //       - Check if OceanTracers correctly select the tracers and the
      //       tracer groups configured in YAML config. file
      //       - Check if OceanTracers correctly identify a tracer or a tracer
      //       group by index and also by name

      // initialize Tracers infrastructure

      Ret = Tracers::init();
      if (Ret != 0) {
         RetVal += 1;
         LOG_ERROR("Tracers: initialzation FAIL");
      }

      // Get group names
      std::vector<std::string> GroupNames = Tracers::getGroupNames();

      // Check if "Base" group exists
      if (std::find(GroupNames.begin(), GroupNames.end(), "Base") !=
          GroupNames.end()) {
         LOG_INFO("Tracers: Group, 'Base', exists PASS");
      } else {
         RetVal += 1;
         LOG_ERROR("Tracers: Group, 'Base', does not exist FAIL");
      }

      // Check if "Debug" group exists
      if (std::find(GroupNames.begin(), GroupNames.end(), "Debug") !=
          GroupNames.end()) {
         LOG_INFO("Tracers: Group, 'Debug', exists PASS");
      } else {
         RetVal += 1;
         LOG_ERROR("Tracers: Group, 'Debug', does not exist FAIL");
      }

      // Check if no more groups for unit testing
      if (GroupNames.size() == 2) {
         LOG_INFO("Tracers: Group size for unit-testing is correct PASS");
      } else {
         RetVal += 1;
         LOG_ERROR("Tracers: Group size for unit-testing is not correct FAIL");
      }

      int TotalLength = 0;

      for (std::string GroupName : GroupNames) {
         std::pair<int, int> GroupRange;
         Ret = Tracers::getGroupRange(GroupName, GroupRange);

         if (Ret != 0) {
            LOG_ERROR("Tracers: getGroupRange returns {} FAIL",
                      std::to_string(Ret));
            RetVal += 1;
         }

         auto [StartIndex, GroupLength] = GroupRange;

         TotalLength += GroupLength;

         // Check if a group contains more than one tracers
         if (GroupLength > 0) {
            LOG_INFO("Tracers: {} tracers retrieval PASS", GroupName);
         } else {
            RetVal += 1;
            LOG_ERROR("Tracers: {} tracers retrieval FAIL", GroupName);
         }

         // Check if tracer index is a member of the Group
         for (int TracerIndex = StartIndex;
              TracerIndex < StartIndex + GroupLength; ++TracerIndex) {
            if (Tracers::isGroupMemberByIndex(TracerIndex, GroupName)) {
               LOG_INFO("Tracers: {} group has the tracer index, {} PASS",
                        GroupName, std::to_string(TracerIndex));
            } else {
               RetVal += 1;
               LOG_ERROR(
                   "Tracers: {} group does not have the tracer index, {} FAIL",
                   GroupName, std::to_string(TracerIndex));
            }
         }

         // Check if tracer index:name mapping is correct
         for (int TracerIndex = StartIndex;
              TracerIndex < StartIndex + GroupLength; ++TracerIndex) {
            std::string TracerName;
            Ret = Tracers::getName(TracerIndex, TracerName);
            if (Ret != 0) {
               LOG_ERROR("Tracers: getName returns {} FAIL",
                         std::to_string(Ret));
               RetVal += 1;
            }

            int RetTracerIndex;
            Ret = Tracers::getIndex(TracerName, RetTracerIndex);
            if (Ret != 0) {
               LOG_ERROR("Tracers: getIndex returns {} FAIL",
                         std::to_string(Ret));
               RetVal += 1;
            }

            if (TracerIndex == RetTracerIndex) {
               LOG_INFO("Tracers: {} group tracer:name mapping for {} is "
                        "correct PASS",
                        GroupName, TracerName);
            } else {
               RetVal += 1;
               LOG_ERROR("Tracers: {} group tracer:name mapping for {} is not "
                         "correct FAIL",
                         GroupName, TracerName);
            }
         }

         // check if tracer field exists
         for (int TracerIndex = StartIndex;
              TracerIndex < StartIndex + GroupLength; ++TracerIndex) {
            auto TracerField = Tracers::getFieldByIndex(TracerIndex);

            if (TracerField) {
               LOG_INFO("Tracers: getFieldByIndex returns a field PASS");
            } else {
               RetVal += 1;
               LOG_ERROR("Tracers: getFieldByIndex returns nullptr FAIL");
            }
         }
      }

      int NAllTracers = Tracers::getNumTracers();

      // Check if total number of tracers is correct
      if (TotalLength == NAllTracers) {
         LOG_INFO("Tracers: getNumTracers() returns correct tracer size PASS");
      } else {
         RetVal += 1;
         LOG_ERROR(
             "Tracers: getNumTracers() returns incorrect tracer size FAIL");
      }

      // set initial values to all tracers for testing
      for (int TimeLevel = 0; TimeLevel + Tracers::NTimeLevels > 0;
           --TimeLevel) {
         for (int TracerIndex = 0; TracerIndex < NAllTracers; ++TracerIndex) {
            auto TrcrArray = Tracers::getByIndex(TimeLevel, TracerIndex);
            if (TrcrArray.data() == nullptr) {
               RetVal += 1;
               LOG_ERROR("Tracers: getByIndex() returns error FAIL");
            }

            parallelFor(
                "initTracer" + std::to_string(TimeLevel) + "-" +
                    std::to_string(TracerIndex),
                {Tracers::NCellsAll, Tracers::NVertLevels},
                KOKKOS_LAMBDA(int Cell, int VertLevel) {
                   TrcrArray(Cell, VertLevel) =
                       RefReal + Cell + VertLevel + TimeLevel + TracerIndex;
                });
         }

         Tracers::copyToHost(TimeLevel);
      }

      // save the original tracers in new arrays
      std::vector<HostArray3DReal> OrgTracerArraysH;

      for (int TimeLevel = 0; TimeLevel + Tracers::NTimeLevels > 0;
           --TimeLevel) {
         OrgTracerArraysH.push_back(Tracers::getAllHost(TimeLevel));
      }

      //
      HostArray3DReal OrgTimeLevel0H = Tracers::getAllHost(0);

      // TODO: 2.5 Requirement: Tracer restart and IO
      //       - save tracers in a file
      //       - Check if OceanTracers correctly read/write tracer data
      //       into/from files
      const std::string TracersFileName = "tracers-unittest.nc";

      Ret = Tracers::saveTracersToFile(TracersFileName, DefDecomp);
      if (Ret == 0) {
         LOG_INFO("Tracers: saveTracersToFile success PASS");
      } else {
         RetVal += 1;
         LOG_ERROR("Tracers: saveTracersToFile failure FAIL");
      }

      // update time lelve
      Tracers::updateTimeLevels();

      // create a tracer arrays for the updated values
      std::vector<HostArray3DReal> UpdatedTracerArraysH;

      for (int TimeLevel = 0; TimeLevel + Tracers::NTimeLevels > 0;
           --TimeLevel) {
         UpdatedTracerArraysH.push_back(Tracers::getAllHost(TimeLevel));
      }

      count = 0;

      // check if Tracers.updateTimeLevels() worked as expected
      for (int TimeIndex = 0; TimeIndex < Tracers::NTimeLevels; ++TimeIndex) {
         int UpdatedTimeIndex = TimeIndex;
         int OrgTimeIndex     = (UpdatedTimeIndex - 1 + Tracers::NTimeLevels) %
                            Tracers::NTimeLevels;

         HostArray3DReal UpdatedTrcrArrayH =
             UpdatedTracerArraysH[UpdatedTimeIndex];
         HostArray3DReal OrgTrcrArrayH = OrgTracerArraysH[OrgTimeIndex];

         for (int TracerIndex = 0; TracerIndex < Tracers::getNumTracers();
              ++TracerIndex) {
            for (int Cell = 0; Cell < Tracers::NCellsAll; Cell++) {
               for (int VertLevel = 0; VertLevel < Tracers::NVertLevels;
                    VertLevel++) {
                  if (UpdatedTrcrArrayH(TracerIndex, Cell, VertLevel) !=
                      OrgTrcrArrayH(TracerIndex, Cell, VertLevel)) {
                     count++;
                  }
               }
            }
         }
      }

      if (count == 0) {
         LOG_INFO(
             "Tracers: All tracer data match after updateTimeLevels() PASS");
      } else {
         RetVal += 1;
         LOG_ERROR("Tracers: Not all tracer data match after "
                   "updateTimeLevels() FAIL");
      }

      // read tracer data from the exported file
      Ret = Tracers::loadTracersFromFile(TracersFileName, DefDecomp);
      if (Ret == 0) {
         LOG_INFO("Tracers: loadTracersFromFile success PASS");
      } else {
         RetVal += 1;
         LOG_ERROR("Tracers: loadTracersFromFile failre FAIL");
      }

      // create history arrays to compare with original arrays

      HostArray3DReal HistoryTimeLevel0H = Tracers::getAllHost(0);

      count = 0;

      for (int TracerIndex=0; TracerIndex < Tracers::getNumTracers(); ++TracerIndex) {
         for (int Cell = 0; Cell < Tracers::NCellsOwned; Cell++) {
            for (int VertLevel = 0; VertLevel < Tracers::NVertLevels; VertLevel++) {
               if (HistoryTimeLevel0H(TracerIndex, Cell, VertLevel) != OrgTimeLevel0H(TracerIndex, Cell, VertLevel)) {
                  count++;
               }
            }
         }
      }


      if (count == 0) {
         LOG_INFO("Tracers: All tracer data match after loadTracersFromFile() PASS");
      } else {
         RetVal += 1;
         LOG_ERROR("Tracers: {} tracer elements didn't match after loadTracersFromFile() FAIL", count);
      }

      // TODO: add more tests on validating each tracer data


      // TODO: 2.7 Requirement: Acceleration or supercycling
      //       - T.B.D.

      // TODO: 2.7 Desired: Per-tracer/group algorithmic requirements
      //       - T.B.D.
      // Finalize Omega objects

      Tracers::clear();

      TimeStepper::clear();
      HorzMesh::clear();
      Decomp::clear();
      MachEnv::removeAll();
      FieldGroup::clear();
      Field::clear();
      Dimension::clear();

      if (RetVal == 0)
         LOG_INFO("Tracers: Successful completion");
   }
   Kokkos::finalize();
   MPI_Finalize();

   if (RetVal >= 256)
      RetVal = 255;

   return RetVal;

} // end of main
//===-----------------------------------------------------------------------===/
