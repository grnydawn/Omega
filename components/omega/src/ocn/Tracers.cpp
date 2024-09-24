//===-- OMEGA Tracers Implementation -----------------------------*- C++
//-*-===//
//
/// \file
/// \brief OMEGA Tracers Implementation
///
//
//===-----------------------------------------------------------------------===/

#include "Tracers.h"
#include "Decomp.h"
#include "IO.h"
#include "Logging.h"
#include "TimeStepper.h"

#include <iostream>

namespace OMEGA {

// Initialize static member variables
std::vector<Array3DR8> Tracers::TracerArrays;
std::vector<HostArray3DR8> Tracers::TracerArraysH;

std::map<std::string, std::pair<int, int>> Tracers::TracerGroups;
std::map<std::string, int> Tracers::TracerIndexes;
std::map<int, std::string> Tracers::TracerNames;
std::vector<std::string> Tracers::TracerDimNames = {"NCells", "NVertLevels"};

Halo *Tracers::MeshHalo = nullptr;

I4 Tracers::NCellsOwned  = 0;
I4 Tracers::NCellsAll    = 0;
I4 Tracers::NCellsSize   = 0;
I4 Tracers::NTimeLevels  = 0;
I4 Tracers::NVertLevels  = 0;
I4 Tracers::CurTimeIndex = 0;
int Tracers::NumTracers  = 0;

//---------------------------------------------------------------------------
// Internal Utilities
//---------------------------------------------------------------------------
std::string Tracers::packTracerFieldName(const std::string &TracerName) {
   return "Tracer" + TracerName;
}

//---------------------------------------------------------------------------
// Initialization
//---------------------------------------------------------------------------
int Tracers::init() {

   int Err = 0;

   // Retrieve mesh cell/edge/vertex totals from Decomp
   HorzMesh *DefHorzMesh = HorzMesh::getDefault();

   NCellsOwned = DefHorzMesh->NCellsOwned;
   NCellsAll   = DefHorzMesh->NCellsAll;
   NCellsSize  = DefHorzMesh->NCellsSize;
   NVertLevels = DefHorzMesh->NVertLevels;

   LOG_INFO("NCellsOwned {}", NCellsOwned);
   LOG_INFO("NCellsAll {}", NCellsAll);
   LOG_INFO("NCellsSize {}", NCellsSize);
   LOG_INFO("NVertLevels {}", NVertLevels);

   MeshHalo = Halo::getDefault();

   auto *DefTimeStepper = TimeStepper::getDefault();
   if (!DefTimeStepper) {
      LOG_ERROR("TimeStepper needs to be initialized before Tracers");
      return -1;
   }
   NTimeLevels = DefTimeStepper->getNTimeLevels();

   if (NTimeLevels < 2) {
      LOG_ERROR("Tracers: the number of time level is lower than 2");
      return -1;
   }

   CurTimeIndex = 0;

   // load Tracers configs
   Config *OmegaConfig = Config::getOmegaConfig();
   Config TracersConfig("Tracers");
   Err = OmegaConfig->get(TracersConfig);
   if (Err != 0) {
      LOG_ERROR("Tracers: Tracers group not found in Config");
      return Err;
   }

   NumTracers      = 0;
   int TracerIndex = 0;

   // get tracers group and tracer names
   for (auto It = TracersConfig.begin(); It != TracersConfig.end(); ++It) {

      int GroupStartIndex = TracerIndex;

      std::string GroupName;
      I4 GroupNameErr = OMEGA::Config::getName(It, GroupName);
      if (GroupNameErr != 0) {
         LOG_ERROR("Tracers: {} tracer group name not found in TracersConfig",
                   GroupName);
         return Err;
      }

      std::vector<std::string> _TracerNames;
      I4 TracerNamesErr = TracersConfig.get(GroupName, _TracerNames);
      if (TracerNamesErr != 0) {
         LOG_ERROR("Tracers: {} group tracers not found in TracersConfig",
                   GroupName);
         return Err;
      }

      for (auto _TracerName : _TracerNames) {
         TracerIndexes[_TracerName] = TracerIndex;
         TracerIndex++;
      }

      TracerGroups[GroupName] =
          std::pair<int, int>(GroupStartIndex, TracerIndex - GroupStartIndex);
      FieldGroup::create("TracerGroup" + GroupName);
   }

   // total number of tracers
   NumTracers = TracerIndex;

   // Initialize tracers arrays for device and host
   TracerArrays.resize(NTimeLevels);
   TracerArraysH.resize(NTimeLevels);

   // Allocate tracers data array and assign to tracers arrays
   for (int TimeIndex = 0; TimeIndex < NTimeLevels; ++TimeIndex) {
      TracerArrays[TimeIndex] =
          Array3DR8("TracerTimeIndex" + std::to_string(TimeIndex), NumTracers,
                    NCellsSize, NVertLevels);
      TracerArraysH[TimeIndex] =
          HostArray3DR8("TracerHTimeIndex" + std::to_string(TimeIndex),
                        NumTracers, NCellsSize, NVertLevels);
   }

// Read tracer definitions from file
#include "TracerDefs.inc"

   // Check if all tracers defined in config file are loaded
   if (TracerIndexes.size() != TracerNames.size()) {
      LOG_ERROR("Tracer: not all tracers defined in config file is loaded.");
      return -1;
   }

   // Add Fields to FieldGroup
   for (const auto &GroupPair : TracerGroups) {
      auto GroupName                   = GroupPair.first;
      std::string TracerFieldGroupName = "TracerGroup" + GroupName;
      auto TracerFieldGroup            = FieldGroup::get(TracerFieldGroupName);

      std::vector<std::string> _TracerNames;
      TracersConfig.get(GroupName, _TracerNames);

      for (auto _TracerName : _TracerNames) {
         std::string TracerFieldName = packTracerFieldName(_TracerName);

         // add tracer Field to field group
         Err = TracerFieldGroup->addField(TracerFieldName);
         if (Err != 0) {
            LOG_ERROR("Error adding {} to field group {}", TracerFieldName,
                      TracerFieldGroupName);
            return -1;
         }

         // Associate Field with data
         int TracerIndex                    = TracerIndexes[_TracerName];
         std::shared_ptr<Field> TracerField = Field::get(TracerFieldName);
         // Create a 2D subview by fixing the first dimension (TracerIndex)
         auto TracerSubview2D = Kokkos::subview(
             TracerArrays[CurTimeIndex], TracerIndex, Kokkos::ALL, Kokkos::ALL);
         Err = TracerField->attachData<Array2DR8>(TracerSubview2D);
         if (Err != 0) {
            LOG_ERROR("Error attaching data array to field {}",
                      TracerFieldName);
            return -1;
         }
      }
   }

   return Err;
}

//---------------------------------------------------------------------------
// Create tracers
//---------------------------------------------------------------------------
int Tracers::define(const std::string &Name, const std::string &Description,
                    const std::string &Units, const std::string &StdName,
                    const R8 ValidMin, const R8 ValidMax, const R8 FillValue) {

   // Do nothing if this tracer is not selected
   if (TracerIndexes.find(Name) == TracerIndexes.end()) {
      return 0;
   }

   auto TracerIndex = TracerIndexes[Name];

   // Return error if tracer already exists
   if (TracerNames.find(TracerIndex) != TracerNames.end()) {
      LOG_ERROR("Tracers: Tracer '{}' already exists", Name);
      return 1;
   }

   // set tracer index to name mapping
   TracerNames[TracerIndex] = Name;

   // create a tracer field
   std::string TracerFieldName = packTracerFieldName(Name);
   auto TracerField = Field::create(TracerFieldName, Description, Units,
                                    StdName, ValidMin, ValidMax, FillValue,
                                    TracerDimNames.size(), TracerDimNames);
   if (!TracerField) {
      LOG_ERROR("Tracers: Tracer field '{}' is not created", TracerFieldName);
      return 1;
   }

   return 0;
}

int Tracers::clear() {

   LOG_INFO("Tracers::clear() called");

   // Deallocate memory for tracer arrays
   TracerArrays.clear();
   TracerArraysH.clear();

   TracerGroups.clear();
   TracerIndexes.clear();
   TracerNames.clear();

   NumTracers  = 0;
   NCellsOwned = 0;
   NCellsAll   = 0;
   NCellsSize  = 0;
   NTimeLevels = 0;
   NVertLevels = 0;

   return 0;
}

//---------------------------------------------------------------------------
// Query tracers
//---------------------------------------------------------------------------

int Tracers::getNumTracers() { return NumTracers; }

int Tracers::getIndex(const std::string &TracerName, int &TracerIndex) {
   // Check if tracer exists
   if (TracerIndexes.find(TracerName) != TracerIndexes.end()) {
      TracerIndex = TracerIndexes[TracerName];
      return 0; // Success
   }

   LOG_ERROR("Tracers: Tracer index for '{}' is not found.", TracerName);
   return 1; // Tracer not found
}

int Tracers::getName(const int TracerIndex, std::string &TracerName) {
   if (TracerNames.find(TracerIndex) != TracerNames.end()) {
      TracerName = TracerNames[TracerIndex];
      return 0; // Success
   }

   LOG_ERROR("Tracers: Tracer name for index '{}' is not found.",
             std::to_string(TracerIndex));
   return 1; // Tracer index not found
}

Array3DR8 Tracers::getAll(const int TimeLevel) {
   if (TimeLevel <= 0 && (TimeLevel + NTimeLevels) > 0) {
      int TimeIndex = (TimeLevel + CurTimeIndex + NTimeLevels) % NTimeLevels;
      return TracerArrays[TimeIndex];
   }

   LOG_ERROR("Tracers: Time index {} is out of range",
             std::to_string(TimeLevel));
   return Array3DR8();
}

Array2DR8 Tracers::getByIndex(const int TimeLevel, const int TracerIndex) {
   // Check if time level is valid
   if (TimeLevel > 0 || (TimeLevel + NTimeLevels) <= 0) {
      LOG_ERROR("Tracers: Time index {} is out of range",
                std::to_string(TimeLevel));
      return Array2DR8();
   }

   // Check if tracer index is valid
   if (TracerIndex < 0 || TracerIndex >= NumTracers) {
      LOG_ERROR("Tracers: Tracer index {} is out of range",
                std::to_string(TracerIndex));
      return Array2DR8();
   }

   int TimeIndex = (TimeLevel + CurTimeIndex + NTimeLevels) % NTimeLevels;
   return Kokkos::subview(TracerArrays[TimeIndex], TracerIndex, Kokkos::ALL,
                          Kokkos::ALL);
}

Array2DR8 Tracers::getByName(const int TimeLevel,
                             const std::string &TracerName) {
   // Check if tracer exists
   if (TracerIndexes.find(TracerName) == TracerIndexes.end()) {
      LOG_ERROR("Tracers: Tracer '{}' does not exist", TracerName);
      return Array2DR8();
   }

   return getByIndex(TimeLevel, TracerIndexes[TracerName]);
}

HostArray3DR8 Tracers::getAllHost(const int TimeLevel) {

   if (TimeLevel <= 0 && (TimeLevel + NTimeLevels) > 0) {

      int TimeIndex = (TimeLevel + CurTimeIndex + NTimeLevels) % NTimeLevels;
      return TracerArraysH[TimeIndex];
   }

   LOG_ERROR("Tracers: Time index {} is out of range",
             std::to_string(TimeLevel));
   return HostArray3DR8();
}

HostArray2DR8 Tracers::getHostByIndex(const int TimeLevel,
                                      const int TracerIndex) {
   // Check if time level is valid
   if (TimeLevel > 0 || (TimeLevel + NTimeLevels) <= 0) {
      LOG_ERROR("Tracers: Time index {} is out of range",
                std::to_string(TimeLevel));
      return HostArray2DR8();
   }

   // Check if tracer index is valid
   if (TracerIndex < 0 || TracerIndex >= NumTracers) {
      LOG_ERROR("Tracers: Tracer index {} is out of range",
                std::to_string(TracerIndex));
      return HostArray2DR8();
   }

   int TimeIndex = (TimeLevel + CurTimeIndex + NTimeLevels) % NTimeLevels;
   return Kokkos::subview(TracerArraysH[TimeIndex], TracerIndex, Kokkos::ALL,
                          Kokkos::ALL);
}

HostArray2DR8 Tracers::getHostByName(const int TimeLevel,
                                     const std::string &TracerName) {
   // Check if tracer exists
   if (TracerIndexes.find(TracerName) == TracerIndexes.end()) {
      LOG_ERROR("Tracers: Tracer '{}' does not exist", TracerName);
      return HostArray2DR8();
   }

   // Get the index of the tracer
   int TracerIndex = TracerIndexes[TracerName];

   return getHostByIndex(TimeLevel, TracerIndex);
}

std::shared_ptr<Field> Tracers::getFieldByName(const std::string &TracerName) {
   // Check if tracer exists
   if (TracerIndexes.find(TracerName) == TracerIndexes.end()) {
      LOG_ERROR("Tracers: Tracer '{}' does not exist", TracerName);
      return nullptr;
   }

   return Field::get(packTracerFieldName(TracerName));
}

std::shared_ptr<Field> Tracers::getFieldByIndex(const int TracerIndex) {
   // Check if tracer index is valid
   if (TracerIndex < 0 || TracerIndex >= NumTracers) {
      LOG_ERROR("Tracers: Tracer index {} is out of range",
                std::to_string(TracerIndex));
      return nullptr;
   }

   return getFieldByName(TracerNames[TracerIndex]);
}

std::vector<std::string> Tracers::getGroupNames() {
   std::vector<std::string> GroupNames;

   for (const auto &GroupPair : TracerGroups) {
      GroupNames.push_back(GroupPair.first);
   }

   return GroupNames;
}

int Tracers::getGroupRange(const std::string &GroupName,
                           std::pair<int, int> &GroupRange) {
   auto it = TracerGroups.find(GroupName);
   if (it != TracerGroups.end()) {
      GroupRange = it->second;
      return 0;
   }

   return -1;
}

bool Tracers::isGroupMemberByIndex(const int TracerIndex,
                                   const std::string GroupName) {
   auto it = TracerGroups.find(GroupName);
   if (it != TracerGroups.end()) {
      int StartIndex  = it->second.first;
      int GroupLength = it->second.second;
      return TracerIndex >= StartIndex &&
             TracerIndex < StartIndex + GroupLength;
   }

   return false;
}

bool Tracers::isGroupMemberByName(const std::string &TracerName,
                                  const std::string &GroupName) {

   int TracerIndex;
   if (getIndex(TracerName, TracerIndex) != 0) {
      return false;
   }

   return isGroupMemberByIndex(TracerIndex, GroupName);
}

int Tracers::copyToDevice(const int TimeLevel) {

   // Check if time index is valid
   if (TimeLevel > 0 || (TimeLevel + NTimeLevels) <= 0) {
      LOG_ERROR("Tracers: Time index {} is out of range",
                std::to_string(TimeLevel));
      return -1;
   }

   int TimeIndex = (TimeLevel + CurTimeIndex + NTimeLevels) % NTimeLevels;
   deepCopy(TracerArrays[TimeIndex], TracerArraysH[TimeIndex]);

   return 0;
}

int Tracers::copyToHost(const int TimeLevel) {

   // Check if time index is valid
   if (TimeLevel > 0 || (TimeLevel + NTimeLevels) <= 0) {
      LOG_ERROR("Tracers: Time index {} is out of range",
                std::to_string(TimeLevel));
      return -1;
   }

   int TimeIndex = (TimeLevel + CurTimeIndex + NTimeLevels) % NTimeLevels;
   deepCopy(TracerArraysH[TimeIndex], TracerArrays[TimeIndex]);

   return 0;
}

int Tracers::exchangeHalo(const int TimeLevel) {
   copyToHost(TimeLevel);

   int TimeIndex = (TimeLevel + CurTimeIndex + NTimeLevels) % NTimeLevels;
   MeshHalo->exchangeFullArrayHalo(TracerArraysH[TimeIndex], OnCell);
   copyToDevice(TimeLevel);
}

int Tracers::updateTimeLevels() {

   // Exchange halo
   exchangeHalo(0);

   // Update TracerField data associations
   for (const auto &TracerPair : TracerIndexes) {
      auto TracerFieldName = packTracerFieldName(TracerPair.first);
      auto TracerIndex     = TracerPair.second;

      std::shared_ptr<Field> TracerField = Field::get(TracerFieldName);

      HostArray2DR8 TracerSubviewH = Kokkos::subview(
          TracerArraysH[CurTimeIndex], TracerIndex, Kokkos::ALL, Kokkos::ALL);
      int Err = TracerField->attachData<HostArray2DR8>(TracerSubviewH);
      if (Err != 0) {
         LOG_ERROR("Error attaching data array to field {}", TracerFieldName);
         return Err;
      }
   }

   CurTimeIndex = (CurTimeIndex + 1) % NTimeLevels;

   return 0;
}

int Tracers::loadTracersFromFile(const std::string &TracerFileName,
                                 Decomp *MeshDecomp) {
   I4 CellDecompR8;
   I4 Err;

   // Create the parallel IO decompositions required to read in state variables
   Err = initParallelIO(CellDecompR8, MeshDecomp);
   if (Err != 0) {
      LOG_CRITICAL("Tracers: error initializing parallel io");
      return -2;
   }

   // Read layerThickness and normalVelocity
   Err = read(TracerFileName, CellDecompR8);
   if (Err != 0) {
      LOG_CRITICAL("Tracers: error reading a file using parallel io");
      return -3;
   }

   // Destroy the parallel IO decompositions
   Err = finalizeParallelIO(CellDecompR8);
   if (Err != 0) {
      LOG_CRITICAL("Tracers: error finalzing parallel io");
      return -4;
   }

   // Sync with device
   copyToDevice(0);

   return 0;
}

int Tracers::saveTracersToFile(const std::string &TracerFileName,
                               Decomp *MeshDecomp) {

   I4 CellDecompR8;
   I4 Err;

   // Sync with device
   copyToHost(0);

   Err = initParallelIO(CellDecompR8, MeshDecomp);
   if (Err != 0) {
      LOG_CRITICAL("Tracers: error initializing parallel io");
      return -6;
   }

   Err = write(TracerFileName, MeshDecomp->NCellsGlobal, CellDecompR8);
   if (Err != 0) {
      LOG_CRITICAL("Tracers: error writing a file using parallel io");
      return -7;
   }

   Err = finalizeParallelIO(CellDecompR8);
   if (Err != 0) {
      LOG_CRITICAL("Tracers: error finalzing parallel io");
      return -8;
   }

   return 0;
}

int Tracers::initParallelIO(I4 &CellDecompR8, Decomp *MeshDecomp) {

   I4 Err;
   I4 NDims             = 3;
   IO::Rearranger Rearr = IO::RearrBox;

   I4 NCellsGlobal = MeshDecomp->NCellsGlobal;

   // Create the IO decomp for arrays with (NCells) dimensions
   std::vector<I4> CellDims{NCellsGlobal, NumTracers, NVertLevels};
   std::vector<I4> CellID(NCellsSize * NumTracers * NVertLevels, -1);

   for (int Cell = 0; Cell < NCellsOwned; ++Cell) {
      I4 CellIDH = MeshDecomp->CellIDH(Cell) - 1;

      for (int Tracer = 0; Tracer < NumTracers; ++Tracer) {
         for (int Level = 0; Level < NVertLevels; ++Level) {

            I4 GlobalID = CellIDH * NumTracers * NVertLevels + Tracer * NVertLevels + Level;
            CellID[Tracer * NCellsSize * NVertLevels + Cell * NVertLevels + Level] = GlobalID;
         }
      }
   }

   Err = IO::createDecomp(CellDecompR8, IO::IOTypeR8, NDims, CellDims,
                          NumTracers * NCellsOwned * NVertLevels, CellID, Rearr);
   if (Err != 0)
      LOG_ERROR("Tracers: IO::createDecomp failed with error code {}", Err);


   return Err;
}

int Tracers::finalizeParallelIO(I4 CellDecompR8) {
   int Err = 0; // default return code

   // Destroy the IO decomp for arrays with (NCells) dimensions
   Err = IO::destroyDecomp(CellDecompR8);
   if (Err != 0)
      LOG_ERROR("Tracers: error destroying cell IO decomposition");

   return Err;
}

// Read Tracers
int Tracers::read(const std::string &TracerFileName, I4 CellDecompR8) {
   I4 Err;
   int TracerFileID;
   int TracerIDCellR8;

   // Open the state file for reading (assume IO has already been initialized)
   Err = IO::openFile(TracerFileID, TracerFileName, IO::ModeRead);
   if (Err != 0) {
      LOG_CRITICAL("Tracers: error opening tracer file");
      return -1;
   }

   Err = IO::readArray(TracerArraysH[CurTimeIndex].data(),
                       NumTracers * NCellsSize * NVertLevels, "TracerArraysH",
                       TracerFileID, CellDecompR8, TracerIDCellR8);
   if (Err != 0)
      LOG_CRITICAL("Tracers: error reading TracerArrays");

   // Finished writing, close file
   Err = IO::closeFile(TracerFileID);
   if (Err != 0) {           
      LOG_ERROR("Tracers: error closing output file");
   }  

   return Err;
}

// Write Tracers
int Tracers::write(const std::string &TracerFileName, int NCellsGlobal, I4 CellDecompR8) {

   I4 Err = 0;
   int TracerFileID;
   R8 FillR8 = -1.23456789e30;

   // Open the state file for reading (assume IO has already been initialized)
   Err = IO::openFile(TracerFileID, TracerFileName, IO::ModeWrite);
   if (Err != 0) {
      LOG_CRITICAL("Tracers: error opening tracer file");
      return -1;
   }

   // Define array dimensions
   int DimTracerID;
   int DimCellID;
   int DimVertID;

   Err = IO::defineDim(TracerFileID, "NCells", NCellsGlobal, DimCellID);
   if (Err != 0) {
      LOG_ERROR("Tracers: error defining Cell dimension FAIL");
      return -3;
   }

   Err = IO::defineDim(TracerFileID, "NumTracers", NumTracers, DimTracerID);
   if (Err != 0) {
      LOG_ERROR("Tracers: error defining tracer dimension FAIL");
      return -2;
   }

   Err = IO::defineDim(TracerFileID, "NVertLevels", NVertLevels, DimVertID);
   if (Err != 0) {
      LOG_ERROR("Tracers: error defining vertical dimension FAIL");
      return -4;
   }

   int TracerDimIDs[3] = {DimCellID, DimTracerID, DimVertID};
   int TracerIDCellR8;

   Err = IO::defineVar(TracerFileID, "TracerArraysH", IO::IOTypeR8, 3,
                       TracerDimIDs, TracerIDCellR8);
   if (Err != 0) {
      LOG_ERROR("Tracers: Error defining TracerArraysH array");
      return -5;
   }

   Err = IO::writeArray(TracerArraysH[CurTimeIndex].data(),
                        NumTracers * NCellsSize * NVertLevels, &FillR8,
                        TracerFileID, CellDecompR8, TracerIDCellR8);
   if (Err != 0) {
      LOG_ERROR("Tracers: error writing TracerArrays");
   }

   // Finished writing, close file
   Err = IO::closeFile(TracerFileID);
   if (Err != 0) {           
      LOG_ERROR("Tracers: error closing output file");
   }  

   return Err;
}

} // namespace OMEGA
