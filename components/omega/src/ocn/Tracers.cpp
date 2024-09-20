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
int Tracers::NumTracers = 0;
std::vector<Array3DReal> Tracers::TracerArrays;
std::vector<HostArray3DReal> Tracers::TracerArraysH;
std::vector<Field> Tracers::TracerFields;
std::map<std::string, std::pair<int, int>> Tracers::TracerGroups;
std::map<std::string, int> Tracers::Index;
std::map<int, std::string> Tracers::Name;
Halo* Tracers::MeshHalo = nullptr;
I4 Tracers::NCellsOwned = 0;
I4 Tracers::NCellsAll = 0;
I4 Tracers::NCellsSize = 0;
I4 Tracers::NTimeLevels = 0;
I4 Tracers::NVertLevels = 0;
I4 Tracers::CurTimeLevel = 0;
I4 Tracers::NewTimeLevel = 0;

//---------------------------------------------------------------------------
// Initialization
//---------------------------------------------------------------------------
int Tracers::init(std::vector<std::string>& GroupNames) {

   // Initialize the tracer infrastructure
   NumTracers = 0;
   TracerArrays.clear();
   TracerArraysH.clear();
   TracerFields.clear();
   TracerGroups.clear();
   TracerIndexes.clear();
   TracerNames.clear();

   // Read tracer definitions from file 

#include "TracerDefs.inc"


   // ...

   NTimeLevels = OMEGA_MAX_TIMELEVELS;

   NCellsOwned = Decomp::getDefault()->getNCellsOwned();
   NCellsAll = Decomp::getDefault()->getNCellsAll();
   NCellsSize = Decomp::getDefault()->getNCellsSize();
   NVertLevels = HorzMesh::getDefault()->getNVertLevels();




}







}
