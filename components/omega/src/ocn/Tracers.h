#ifndef OMEGA_TRACERS_H
#define OMEGA_TRACERS_H
//===-- ocn/Tracers.h - tracers --------------------*- C++ -*-===//
//
/// \file
/// \brief Contains the tracers for an OMEGA sub-domain.
///
/// NOTES: - Once tracers are initialized, no updates are expected.
///        - Support for multi-threading is not considered.
///        - While Tracers is implemented as a class, the class
///          acts as a singleton because tracers are unchanged during execution
//===----------------------------------------------------------------------===//

#include "DataTypes.h"
#include "Decomp.h"
#include "Halo.h"
#include "HorzMesh.h"
#include "MachEnv.h"

#include <string>

namespace OMEGA {

    /// A class for the tracers variable information

/// The Tracers class provides a container for the groups of trace
/// variables. It contains methods which handle IO and time level updates.
class Tracers {

 private:
   static int NumTracers;

   // static storage of the tracer arrays
   static std::vector<Array3DR8> TracerArrays;      ///< TimeLevels -> [Tracer, Cell, Vert]
   static std::vector<HostArray3DR8> TracerArraysH; ///< TimeLevels -> [Tracer, Cell, Vert]

   // maps for managing tracer groups
   // Key of this map is a group name and
   // Value is a pair of GroupStartIndex and GroupLength
   static std::map<std::string, std::pair<int, int>> TracerGroups;

   // maps for matching tracer names with indices (both directions)
   static std::map<std::string, int> TracerIndexes;
   static std::map<int, std::string> TracerNames;

   static int initParallelIO(
      I4 &CellDecompR8,
      Decomp *MeshDecomp
   );

   static int finalizeParallelIO(
      I4 CellDecompR8
   );

   static int read(
      int TracersFileID,
      I4 CellDecompR8
   );

   static Halo *MeshHalo;

   static I4 NCellsOwned; ///< Number of cells owned by this task
   static I4 NCellsAll;   ///< Total number of local cells (owned + all halo)
   static I4 NCellsSize;  ///< Array size (incl padding, bndy cell) for cell arrays

   const std::vector<std::string> TracerDimNames = {"NCells", "NVertLevels"}; // Tracer dimension names

   static I4 NTimeLevels; ///< Number of time levels in tracer variable arrays
   static I4 NVertLevels; ///< Number of vertical levels in tracer variable arrays
   static I4 CurTimeLevel; ///< Time dimension index for current level

   // pack tracer field name 
   static std::string packTracerFieldName(const std::string &TracerName);

   // locally defines all tracers but do not allocates memory
   static int define(
      const std::string &Name,        ///< [in] Name of tracer
      const std::string &Description, ///< [in] Long name or description
      const std::string &Units,       ///< [in] Units
      const std::string &StdName,     ///< [in] CF standard Name
      const R8 ValidMin,            ///< [in] min valid field value
      const R8 ValidMax,            ///< [in] max valid field value
      const R8 FillValue            ///< [in] value for undef entries
   );

 public:
   //---------------------------------------------------------------------------
   // Initialization
   //---------------------------------------------------------------------------
   /// read tracer defintions, allocate tracer arrays and initializes the tracers
   static int init();

   /// deallocates tracer arrays
   static int clear();

   //---------------------------------------------------------------------------
   // Query tracers
   //---------------------------------------------------------------------------

   static int getNumTracers();

   static int getIndex(
      const std::string &TracerName, ///< [in] tracer name
      int &TracerIndex               ///< [out] tracer index
   );

   static int getName(
      const int TracerIndex,         ///< [in] tracer index
      std::string &TracerName        ///< [out] tracer name
   );

   // returns all tracer device arrays. If it does not exist, return nullptr
   static Array3DR8 getAll(
      const int TimeLevel            ///< [in] time level index
   );

   // returns a tracer device array by tracer index. If it does not exist, return nullptr
   static Array2DR8 getByIndex(
      const int TimeLevel            ///< [in] time level index
      const int TracerIndex,         ///< [in] global tracer index
   );

   // returns a tracer device array by tracer name. If it does not exist, return nullptr
   static Array2DR8 getByName(
      const int TimeLevel            ///< [in] time level index
      const std::string &TracerName, ///< [in] global tracer name
   );

   // returns all tracer device arrays. If it does not exist, return nullptr
   static HostArray3DR8 getAllHost(
      const int TimeLevel            ///< [in] time level index
   );

   // returns a tracer host array by tracer index. If it does not exist, return nullptr
   static HostArray2DR8 getHostByIndex(
      const int TimeLevel            ///< [in] time level index
      const int TracerIndex,         ///< [in] global tracer index
   );

   // returns a tracer host array by tracer name. If it does not exist, return nullptr
   static HostArray2DR8 getHostByName(
      const int TimeLevel            ///< [in] time level index
      const std::string &TracerName, ///< [in] global tracer name
   );

   // returns a field by tracer index. If it does not exist, return nullptr
   static std::shared_ptr<Field> getFieldByIndex(
      const int TracerIndex ///< [in] global tracer index
   );

   // returns a field by tracer name. If it does not exist, return nullptr
   static std::shared_ptr<Field> getFieldByName(
      const std::string &TracerName ///< [in] global tracer name
   );

   //---------------------------------------------------------------------------
   // Tracer group query
   //---------------------------------------------------------------------------

   // return a vector of group names.
   static std::vector<std::string> getGroupNames();

   // get a pair of (group start index, group length)
   static int getGroupRange(
      const std::string &GroupName     ///< [in] group name 
      std::pair<int, int> &GroupRange  ///< [out] group range
   );

   // check if a tracer is a member of group by tracer index
   static bool isGroupMemberByIndex(
      const int TracerIndex,       ///< [in] tracer index
      const std::string GroupName  ///< [in] group name 
   );

   // check if a tracer is a member of group by tracer name
   static bool isGroupMemberByName(
      const std::string &TracerName,   ///< [in] global tracer name
      const std::string &GroupName     ///< [in] group name
   );

   //---------------------------------------------------------------------------
   // File-IO support
   //---------------------------------------------------------------------------

   /// load tracers from file
   static int loadTracersFromFile(
      const std::string &TracersFileName, ///< [in] tracer file name
      Decomp *MeshDecomp                  ///< [in] mesh decompositon
   );

   /// save tracers to file
   static int saveTracersToFile(
      const std::string &TracersFileName, ///< [in] tracer file name
      Decomp *MeshDecomp                  ///< [in] mesh decompositon
   );

   //---------------------------------------------------------------------------
   // Halo exchange and update time level
   //---------------------------------------------------------------------------

   /// Exchange halo
   static int exchangeHalo(
      const int TimeLevel ///< [in] tracer time level
   );

   /// Swap time levels to update tracer arrays
   static int updateTimeLevels();

   //---------------------------------------------------------------------------
   // Device-Host data movement
   //---------------------------------------------------------------------------

   /// Copy tracers variables from host to device
   static int copyToDevice(
      const int TimeLevel ///< [in] tracer time level
   );

   /// Copy tracers variables from device to host
   static int copyToHost(
      const int TimeLevel ///< [in] tracer time level
   );

   //---------------------------------------------------------------------------
   // Forbid copy and move construction
   //---------------------------------------------------------------------------

   Tracers(const Tracers &) = delete;
   Tracers(Tracers &&)      = delete;

}; // end class Tracers

} // end namespace OMEGA

//===----------------------------------------------------------------------===//
#endif // defined OMEGA_TRACERS_H
