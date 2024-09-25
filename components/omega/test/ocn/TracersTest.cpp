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

const R8 RefR8 = 3.0;
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

      int NTracers = Tracers::getNumTracers();
      int NCellsOwned = Tracers::NCellsOwned;
      int NCellsAll = Tracers::NCellsAll;
      int NCellsSize = Tracers::NCellsSize;
      int NVertLevels = Tracers::NVertLevels;

      // Check if total number of tracers is correct
      if (TotalLength == NTracers) {
         LOG_INFO("Tracers: getNumTracers() returns correct tracer size PASS");
      } else {
         RetVal += 1;
         LOG_ERROR(
             "Tracers: getNumTracers() returns incorrect tracer size FAIL");
      }

      // Referecne host array of current time level for later tests
      HostArray3DR8 RefHostArray = HostArray3DR8("RefHostArray", NTracers, NCellsSize, NVertLevels);

      // intialize tracer elements of all time levels
      for (int TimeLevel = 0; TimeLevel + Tracers::NTimeLevels > 0; --TimeLevel) {
         HostArray3DR8 TempHostArray = Tracers::getAllHost(TimeLevel);
         for (int Tracer = 0; Tracer < NTracers; ++Tracer) {
            for (int Cell = 0; Cell < NCellsSize; Cell++) {
               for (int Vert= 0; Vert< NVertLevels; Vert++) {
                    TempHostArray(Tracer, Cell, Vert) = RefR8 + Tracer + Cell + Vert + TimeLevel;
                    if (TimeLevel == 0)
                       RefHostArray(Tracer, Cell, Vert) = TempHostArray(Tracer, Cell, Vert);
               }
            }
         }
         Tracers::copyToDevice(TimeLevel);
      }

      // Reference field vector of all tracers
      std::vector<std::shared_ptr<Field>> RefFields;

      // Reference field data of all tracers
      std::vector<Array2DR8> RefFieldDataArray;

      // get field references of all tracers
      for (int Tracer = 0; Tracer < NTracers; ++Tracer) {
         auto TracerField = Tracers::getFieldByIndex(Tracer);
         RefFields.push_back(TracerField);
         RefFieldDataArray.push_back(TracerField->getDataArray<Array2DR8>());
      }

      // update time levels
      Tracers::updateTimeLevels();

      // check if time level shift works
      // Previous time level(-1) should match to RefHostArray elements
      Array3DR8 PrevArray = Tracers::getAll(-1);
      count = -1;

      parallelReduce(
         "reduce1", {NTracers, NCellsOwned, NVertLevels},
         KOKKOS_LAMBDA(int Tracer, int Cell, int Vert, int &Accum) {
            if (std::abs(PrevArray(Tracer, Cell, Vert) - (RefR8 + Tracer + Cell + Vert)) > 1e-9) {
               Accum++;
            }
         },
         count);

      if (count == 0) {
         LOG_INFO(
             "Tracers: Tracer data match after updateTimeLevels() PASS");
      } else {
         RetVal += 1;
         LOG_ERROR("Tracers: Not all tracer data match after "
                   "updateTimeLevels():{} FAIL", count);
      }

      // test field data 
      for (int Tracer = 0; Tracer < NTracers; ++Tracer) {
         auto TracerField = Tracers::getFieldByIndex(Tracer);
         Array2DR8 TestFieldData = TracerField->getDataArray<Array2DR8>();
         Array2DR8 RefFieldData = RefFieldDataArray[Tracer];


         count = -1;

               //if (std::abs(RefFieldData[Tracer](Cell, Vert) - TestFieldData(Cell, Vert)) > 1e-9) {
         parallelReduce(
            "reduce2", {NCellsOwned, NVertLevels},
            KOKKOS_LAMBDA(int Cell, int Vert, int &Accum) {
               if (std::abs(RefFieldData(Cell, Vert) - TestFieldData(Cell, Vert)) > 1e-9) {
                  Accum++;
               }
            },
            count);

   
         if (count > 0) {
            LOG_INFO(
                "Tracers: Tracer field data correctly catch the difference after updateTimeLevels() PASS");
         } else {
            RetVal += 1;
            LOG_ERROR("Tracers: Tracer field data should not match after updateTimeLevels() FAIL");
         }
      }

      // update time levels to cycle back to original index
      for (int TimeLevel = -1; TimeLevel + Tracers::NTimeLevels > 0; --TimeLevel) {
         // update time levels
         Tracers::updateTimeLevels();
      }

      // test field data 
      for (int Tracer = 0; Tracer < NTracers; ++Tracer) {
         auto TracerField = Tracers::getFieldByIndex(Tracer);
         Array2DR8 TestFieldData = TracerField->getDataArray<Array2DR8>();
         Array2DR8 RefFieldData = RefFieldDataArray[Tracer];


         count = -1;

               //if (std::abs(RefFieldData[Tracer](Cell, Vert) - TestFieldData(Cell, Vert)) > 1e-9) {
         parallelReduce(
            "reduce3", {NCellsOwned, NVertLevels},
            KOKKOS_LAMBDA(int Cell, int Vert, int &Accum) {
               if (std::abs(RefFieldData(Cell, Vert) - TestFieldData(Cell, Vert)) > 1e-9) {
                  Accum++;
               }
            },
            count);

   
         if (count == 0) {
            LOG_INFO(
                "Tracers: Tracer field data correctly match after updateTimeLevels() back to original index PASS");
         } else {
            RetVal += 1;
            LOG_ERROR("Tracers: Not all tracer data match after updateTimeLevels() back to original index FAIL");

         }
      }

      // Test host array of current time level for
      count = 0;

      // intialize tracer elements of all time levels
      for (int Tracer = 0; Tracer < NTracers; ++Tracer) {
         HostArray2DR8 TestHostArray = Tracers::getHostByIndex(0, Tracer);
         for (int Cell = 0; Cell < NCellsOwned; Cell++) {
            for (int Vert= 0; Vert< NVertLevels; Vert++) {
               if (std::abs(RefHostArray(Tracer, Cell, Vert) - TestHostArray(Cell, Vert)) > 1e-9)
                  ++count;
            }
         }
      }

      if (count == 0) {
         LOG_INFO(
             "Tracers: Tracer getHostByIndex correctly retreive tracer data PASS");
      } else {
         RetVal += 1;
         LOG_Error( "Tracers: Tracer getHostByIndex retreives incorrect tracer data FAIL");
      }

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
