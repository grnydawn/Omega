/===-- OMEGA Tracers Implementation -----------------------------*- C++ -*-===//
//
/// \file
/// \brief OMEGA Tracers Implementation
///
//
//===-----------------------------------------------------------------------===/

#include "Tracers.h"

namespace OMEGA {

// Initialize static member variables
std::vector<Array3DR8> Tracers::TracerArrays;
std::vector<HostArray3DR8> Tracers::TracerArraysH;

std::map<std::string, std::pair<int, int>> Tracers::TracerGroups;
std::map<std::string, int> Tracers::TracerIndexes;
std::map<int, std::string> Tracers::TracerNames;

Halo* Tracers::MeshHalo = nullptr;

I4 Tracers::NCellsOwned = 0;
I4 Tracers::NCellsAll = 0;
I4 Tracers::NCellsSize = 0;
I4 Tracers::NTimeLevels = 0;
I4 Tracers::NVertLevels = 0;
I4 Tracers::CurTimeLevel = 0;

//---------------------------------------------------------------------------
// Internal Utilities
//---------------------------------------------------------------------------

static std::string packTracerFieldName(const std::string &TracerName) {
   return "Tracer" + TracerName;
}

//---------------------------------------------------------------------------
// Initialization
//---------------------------------------------------------------------------
int Tracers::init() {

   HorzMesh *DefHorzMesh = HorzMesh::getDefault();
   NVertLevels = DefHorzMesh->NVertLevels;

   auto *DefTimeStepper = TimeStepper::getDefault();
   if (!DefTimeStepper) {
      LOG_ERROR("TimeStepper needs to be initialized before OceanState");
   }

   MeshHalo = Halo::getDefault();

   auto *DefTimeStepper = TimeStepper::getDefault();
   NTimeLevels = DefTimeStepper->getNTimeLevels();

   if (NTimeLevels < 2) {
      LOG_CRITICAL("Tracers: the number of time level is lower than 2");
      return Err;
   }

   CurTimeLevel = 0;

   NCellsOwned = Decomp::getDefault()->getNCellsOwned();
   NCellsAll = Decomp::getDefault()->getNCellsAll();
   NCellsSize = Decomp::getDefault()->getNCellsSize();
   NVertLevels = HorzMesh::getDefault()->getNVertLevels();

   // load Tracers configs
   Config *OmegaConfig = Config::getOmegaConfig();
   Config TracersConfig("Tracers");
   Err = OmegaConfig->get(TracersConfig);
   if (Err != 0) {
      LOG_CRITICAL("Tracers: Tracers group not found in Config");
      return Err;
   }

   NumTracers = 0;
   int TracerIndex = 0;

   // get tracers group and tracer names
   for (auto It = TracersConfig->begin(); It != TracersConfig->end(); ++It) {

      int GroupStartIndex = TracerIndex;

      std::string GroupName;
      I4 GroupNameErr = OMEGA::Config::getName(It, GroupName);
      if (GroupNameErr != 0) {
         LOG_CRITICAL("Tracers: %s tracer group name not found in TracersConfig", GroupName);
         return Err;
      }  

      std::vector<std::string> TracerGroupNames;
      I4 TracerNamesErr = TracersConfig->get(GroupName, TracerGroupNames);
      if (TracerNamesErr != 0) {
         LOG_CRITICAL("Tracers: %s group tracers not found in TracersConfig", GroupName);
         return Err;
      }  

      for (TracerGroupName: TracerGroupNames) {
         TracerIndexes[TracerGroupName] = TracerIndex;
         TracerIndex++;
      }

      TracerGroups[GroupName] = std::pair<int, int>(GroupStartIndex, TracerIndex - GroupStartIndex);
      FieldGroup::create("TracerGroup" + GroupName);
   }

   // total number of tracers
   NumTracers = TracerIndex;

   // Initialize tracers arrays for device and host
   TracerArrays.resize(NTimeLevels);
   TracerArraysH.resize(NTimeLevels);

   // Allocate tracers data array and assign to tracers arrays
   for (int TimeLevel = 0; TimeLevel < NTimeLevels; ++TimeLevel) {
      TracerArrays[TimeLevel] = Array3DR8("TracerTimeLevel" + std::to_string(TimeLevel), NumTracers, NCellsSize, NVertLevels);
      TracerArraysH[TimeLevel] = HostArray3DR8("TracerHTimeLevel" + std::to_string(TimeLevel), NumTracers, NCellsSize, NVertLevels);
   }

   // Read tracer definitions from file
   #include "TracerDefs.inc"

   // Check if all tracers defined in config file are loaded
   if (TracerIndexes.size() != TracerNames.size()) {
      LOG_ERROR("Tracer: not all tracers defined in config file is loaded.");
      return 1;
   }

   // Add Fields to FieldGroup
   for (GroupName: GroupNames) {
      std::string TracerFieldGroupName = "TracerGroup" + GroupName
      TracerFieldGroup = FieldGroup::get(TracerFieldGroupName);

      std::vector<std::string> TracerGroupNames;
      TracersConfig->get(GroupName, TracerGroupNames);

      for (TracerGroupName: TracerGroupNames) {
         std::string TracerFieldName = packTracerFieldName(TracerGroupName)

         // add tracer Field to field group
         Err = TracerFieldGroup->addField(TracerFieldName);
         if (Err != 0)
            LOG_ERROR("Error adding {} to field group {}", TracerFieldName,
                      TracerFieldGroupName);

         // Associate Field with data
         int TracerIndex = TracerIndexes[TracerGroupName];
         std::shared_ptr<Field> TracerField = Field::get(TracerFieldName);
         Err = TracerField->attachData<Array2DR8>(TracerArrays[CurTimeLevel][TracerIndex]);
         if (Err != 0)
            LOG_ERROR("Error attaching data array to field {}", TracerFieldName);
      }
   }
}

//---------------------------------------------------------------------------
// Create tracers
//---------------------------------------------------------------------------
int Tracers::define(
   const std::string& Name,
   const std::string& Description,
   const std::string& Units,
   const std::string& StdName,
   const R8 ValidMin,
   const R8 ValidMax,
   const R8 FillValue
) {

   // Do nothing if this tracer is not selected
   if (TracerIndexes.find(Name) == TracerIndexes.end()) {
      return 0;
   }

   TracerIndex = TracerIndexes[Name];

   // Return error if tracer already exists
   if (TracerNames.find(TracerIndex) != TracerIndexes.end()) {
      LOG_ERROR("Tracers: Tracer '%s' already exists", Name);
      return 1;
   }

   // set tracer index to name mapping
   TracerNames[TracerIndex] = Name;

   // create a tracer field
   std::string TracerFieldName = packTracerFieldName(Name);
   auto TracerField = Field::create(TracerFieldName, Description, Units, StdName, ValidMin, ValidMax, FillValue, TracerDimNames.size(), TracerDimNames);
   if (!TracerField) {
      LOG_ERROR("Tracers: Tracer field '%s' is not created", TracerFieldName);
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

   NumTracers = 0;
   NCellsOwned = 0;
   NCellsAll = 0;
   NCellsSize = 0;
   NTimeLevels = 0;
   NVertLevels = 0;

   return 0;
}

//---------------------------------------------------------------------------
// Query tracers
//---------------------------------------------------------------------------

int Tracers::getNumTracers() {
   return NumTracers;
}

int Tracers::getIndex(const std::string& TracerName, int& TracerIndex) {
   // Check if tracer exists
   if (TracerIndexes.find(TracerName) != TracerIndexes.end()) {
      TracerIndex = TracerIndexes[TracerName];
      return 0; // Success
   }

   LOG_ERROR("Tracers: Tracer index for '%s' is not found.", TracerName);
   return 1; // Tracer not found
}

int Tracers::getName(const int TracerIndex, std::string& TracerName) {
   if (Name.find(TracerIndex) != Name.end()) {
      TracerName = Name[TracerIndex];
      return 0; // Success
   }

   LOG_ERROR("Tracers: Tracer name for index '%d' is not found.", TracerIndex);
   return 1; // Tracer index not found
}

Array3DReal Tracers::getAll(const int TimeLevel) {
   if (TimeLevel >= 0 && TimeLevel < NTimeLevels) {
      return TracerArrays[TimeLevel];
   }

   LOG_ERROR("Tracers: Time index %d is out of range", TimeLevel);
   return nullptr;
}

Array2dReal Tracers::getByIndex(const int TimeLevel, const int TracerIndex) {
   // Check if time index is valid
   if (TimeLevel < 0 || TimeLevel >= NTimeLevels) {
      LOG_ERROR("Tracers: Time index %d is out of range", TimeLevel);
      return nullptr;
   }

   // Check if tracer index is valid
   if (TracerIndex < 0 || TracerIndex >= NumTracers) {
      LOG_ERROR("Tracers: Tracer index %d is out of range", TracerIndex);
      return nullptr;
   }

   return TracerArrays[TimeLevel][TracerIndex];

}

Array2dReal Tracers::getByName(const int TimeLevel, const std::string& TracerName) {
   // Check if tracer exists
   if (TracerIndexes.find(TracerName) == TracerIndexes.end()) {
      LOG_ERROR("Tracers: Tracer '%s' does not exist", TracerName);
      return nullptr;
   }

   // Get the index of the tracer
   int TracerIndex = TracerIndexes[TracerName];

   return getByIndex(TimeLevel, TracerIndex);
}

HostArray3DReal Tracers::getAllHost(const int TimeLevel) {
   if (TimeLevel >= 0 && TimeLevel < NTimeLevels) {
      return TracerArraysH[TimeLevel];
   }

   LOG_ERROR("Tracers: Time index %d is out of range", TimeLevel);
   return nullptr;
}

HostArray2dReal Tracers::getHostByIndex(const int TimeLevel, const int TracerIndex) {
   // Check if time index is valid
   if (TimeLevel < 0 || TimeLevel >= NTimeLevels) {
      LOG_ERROR("Tracers: Time index %d is out of range", TimeLevel);
      return nullptr;
   }

   // Check if tracer index is valid
   if (TracerIndex < 0 || TracerIndex >= NumTracers) {
      LOG_ERROR("Tracers: Tracer index %d is out of range", TracerIndex);
      return nullptr;
   }

   return TracerArraysH[TimeLevel][TracerIndex];

}

HostArray2dReal Tracers::getHostByName(const int TimeLevel, const std::string& TracerName) {
   // Check if tracer exists
   if (TracerIndexes.find(TracerName) == TracerIndexes.end()) {
      LOG_ERROR("Tracers: Tracer '%s' does not exist", TracerName);
      return nullptr;
   }

   // Get the index of the tracer
   int TracerIndex = TracerIndexes[TracerName];

   return getHostByIndex(TimeLevel, TracerIndex);
}

static std::shared_ptr<Field> Tracers::getFieldByName(const std::string& TracerName) {
   // Check if tracer exists
   if (TracerIndexes.find(TracerName) == TracerIndexes.end()) {
      LOG_ERROR("Tracers: Tracer '%s' does not exist", TracerName);
      return nullptr;
   }

   return Field::get(packTracerFieldName(TracerName));
}


static std::shared_ptr<Field> Tracers::getFieldByIndex(const int TracerIndex) {
   // Check if tracer index is valid
   if (TracerIndex < 0 || TracerIndex >= NumTracers) {
      LOG_ERROR("Tracers: Tracer index %d is out of range", TracerIndex);
      return nullptr;
   }

   return getFieldByName(TracerNames[TracerIndex]);
}


std::vector<std::string> Tracers::getGroupNames() {
   std::vector<std::string> GroupNames;

   for (const auto& GroupPair : TracerGroups) {
      GroupNames.push_back(GroupPair.first);
   }

   return GroupNames;
}

int Tracers::getGroupRange(const std::string& GroupName, std::pair<int, int>& GroupRange) {
   auto it = TracerGroups.find(GroupName);
   if (it != TracerGroups.end()) {
      GroupRange = it->second;
      return 0;
   }

   return -1;
}

bool Tracers::isGroupMemberByIndex(const int TracerIndex, const std::string GroupName) {
   auto it = TracerGroups.find(GroupName);
   if (it != TracerGroups.end()) {
      int StartIndex = it->second.first;
      int GroupLength = it->second.second;
      return TracerIndex >= StartIndex && TracerIndex < StartIndex + GroupLength;
   }

   return false;

}

bool Tracers::isGroupMemberByName(const std::string& TracerName, const std::string& GroupName) {

   int TracerIndex;
   if (getIndex(TracerName, TracerIndex) != 0) {
      return false;
   }

   return isGroupMemberByIndex(TracerIndex, GroupName);
}

int Tracers::copyToDevice(const int TimeLevel) {

   // Check if time index is valid
   if (TimeLevel < 0 || TimeLevel >= NTimeLevels) {
      LOG_ERROR("Tracers: Time index %d is out of range", TimeLevel);
      return -1;
   }
  
   deepCopy(TracerArrays[TimeLevel], TracerArraysH[TimeLevel]);

   return 0;
}
   
int Tracers::copyToHost(const int TimeLevel) {

   // Check if time index is valid
   if (TimeLevel < 0 || TimeLevel >= NTimeLevels) {
      LOG_ERROR("Tracers: Time index %d is out of range", TimeLevel);
      return -1;
   }

   deepCopy(TracerArraysH[TimeLevel], TracerArrays[TimeLevel]);

   return 0;
}


int Tracers::exchangeHalo(const int TimeLevel) {
   copyToHost(TimeLevel);
   MeshHalo->exchangeFullArrayHalo(TracerArraysH[TimeLevel], OnCell);
   copyToDevice(TimeLevel);
}

static int updateTimeLevels() {

   // Exchange halo
   exchangeHalo(CurTimeLevel);

   // Update TracerField data associations
   for (const auto& TracerPair : TracerIndexes) {
      TracerFieldName = packTracerFieldName(TracerPair.first);
      TracerIndex = TracerPair.second;

      std::shared_ptr<Field> TracerField = Field::get(TracerFieldName);
      Err = TracerField->attachData<Array2DR8>(TracerArrays[CurTimeLevel][TracerIndex]);
      if (Err != 0)
         LOG_ERROR("Error attaching data array to field {}", TracerFieldName);
   }

   CurTimeLevel = (CurTimeLevel + 1) % NTimeLevels;
}

int Tracers::loadTracersFromFile(const std::string& TracerFileName, Decomp* MeshDecomp) {
   int TracerFileID;
   I4 CellDecompR8;
   I4 Err;

   // Open the state file for reading (assume IO has already been initialized)
   Err = IO::openFile(TracerFileID, TracerFileName, IO::ModeRead);
   if (Err != 0) {
      LOG_CRITICAL("Tracers: error opening tracer file");
      return -1;
   }

   // Create the parallel IO decompositions required to read in state variables
   Err = initParallelIO(CellDecompR8, MeshDecomp);
   if (Err != 0) {
      LOG_CRITICAL("Tracers: error initializing parallel io");
      return -1;
   }

   // Read layerThickness and normalVelocity
   Err = read(TracerFileID, CellDecompR8);
   if (Err != 0) {
      LOG_CRITICAL("Tracers: error reading a file using parallel io");
      return -1;
   }

   // Destroy the parallel IO decompositions
   Err = finalizeParallelIO(CellDecompR8);
   if (Err != 0) {
      LOG_CRITICAL("Tracers: error finalzing parallel io");
      return -1;
   }

   // Sync with device
   copyToDevice(CurTimeLevel);

   return 0;
}

int Tracers::saveTracersToFile(const std::string& TracersFileName, Decomp* MeshDecomp) {
   int TracerFileID;
   I4 CellDecompR8;
   I4 Err;

   // Sync with device
   copyToHost(CurTimeLevel);

   // Open the state file for reading (assume IO has already been initialized)
   Err = IO::openFile(TracerFileID, TracerFileName, IO::ModeWrite);
   if (Err != 0) {
      LOG_CRITICAL("Tracers: error opening tracer file");
      return -1;
   }

   // Create the parallel IO decompositions required to read in state variables
   Err = initParallelIO(CellDecompR8, MeshDecomp);
   if (Err != 0) {
      LOG_CRITICAL("Tracers: error initializing parallel io");
      return -1;
   }

   // Read layerThickness and normalVelocity
   Err = write(TracerFileID, CellDecompR8);
   if (Err != 0) {
      LOG_CRITICAL("Tracers: error writing a file using parallel io");
      return -1;
   }

   // Destroy the parallel IO decompositions
   Err = finalizeParallelIO(CellDecompR8);
   if (Err != 0) {
      LOG_CRITICAL("Tracers: error finalzing parallel io");
      return -1;
   }

   return 0;
}

// Initialize the parallel IO decompositions for the mesh variables
int Tracers::initParallelIO(I4 &CellDecompR8, Decomp *MeshDecomp) {
   I4 Err;
   I4 NDims             = 3;
   IO::Rearranger Rearr = IO::RearrBox;

   // Create the IO decomp for arrays with (NCells) dimensions
   std::vector<I4> CellDims{1, MeshDecomp->NCellsGlobal, NVertLevels};
   std::vector<I4> CellID(NCellsAll * NVertLevels, -1);

   for (int Cell = 0; Cell < NCellsAll; ++Cell) {
      for (int Level = 0; Level < NVertLevels; ++Level) {
         I4 GlobalID = (MeshDecomp->CellIDH(Cell) - 1) * NVertLevels + Level;
         CellID[Cell * NVertLevels + Level] = GlobalID;
      }
   }

   Err = IO::createDecomp(CellDecompR8, IO::IOTypeR8, NDims, CellDims,NCellsAll * NVertLevels, CellID, Rearr);

   return Err;
}


// Destroy parallel decompositions
int Tracers::finalizeParallelIO(I4 CellDecompR8) {
   int Err = 0; // default return code

   // Destroy the IO decomp for arrays with (NCells) dimensions
   Err = IO::destroyDecomp(CellDecompR8);
   if (Err != 0)
      LOG_CRITICAL("Tracers: error destroying cell IO decomposition");

   return Err;
}

// Read Ocean Tracer
int Tracers::read(int TracerFileID, I4 CellDecompR8) {
   I4 Err;

   // Read LayerThickness
   int LayerThicknessID;

   Err = IO::readArray(LayerThicknessH[CurLevel].data(), NCellsAll,"layerThickness", TracerFileID, CellDecompR8,LayerThicknessID);
   if (Err != 0)
      LOG_CRITICAL("Tracers: error reading layerThickness");

   return Err;
}



} // OMEGA namespace
