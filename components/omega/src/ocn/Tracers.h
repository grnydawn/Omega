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

#if !defined(OMEGA_MAX_TIMELEVELS)
#define OMEGA_MAX_TIMELEVELS 5
#endif

#define DEFINE_TRACER(NAME, ...) \
    if (SelectedTracers.find(NAME) != SelectedTracers.end()) \
        OMEGA::Tracers::define(NAME, __VA_ARGS__);

namespace OMEGA {

    /// A class for the tracers variable information

/// The Tracers class provides a container for the groups of trace
/// variables. It contains methods which handle IO and time level updates.
class Tracers {

 private:
   static int NumTracers;

   // static storage of the tracer arrays and fields
   static std::vector<Array3DReal> TracerArrays;
   static std::vector<HostArray3DReal> TracerArraysH;
   static std::vector<Field> TracerFields;

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

   static const I4 MaxTimeLevels = OMEGA_MAX_TIMELEVELS; ///< Maximum number of time levels

   static I4 NTimeLevels; ///< Number of time levels in tracer variable arrays
   static I4 NVertLevels; ///< Number of vertical levels in tracer variable arrays

   static I4 CurTimeLevel; ///< Time dimension index for current level
   static I4 NewTimeLevel; ///< Time dimension index for new level

 public:
   //---------------------------------------------------------------------------
   // Initialization
   //---------------------------------------------------------------------------
   /// read tracer defintions, allocate tracer arrays and initializes the tracers
   static int init(
      std::vector<std::string>& GroupNames = std::vector<std::string>{}
   );

   /// deallocates tracer arrays
   static int clear();

   //---------------------------------------------------------------------------
   // Create tracers
   //---------------------------------------------------------------------------

   // locally defines all tracers but do not allocates memory
   static int define(
      const std::string &Name,        ///< [in] Name of tracer
      const std::string &Description, ///< [in] Long name or description
      const std::string &Units,       ///< [in] Units
      const std::string &StdName,     ///< [in] CF standard Name
      OMEGA::Real ValidMin,           ///< [in] min valid field value
      OMEGA::Real ValidMax,           ///< [in] max valid field value
      OMEGA::Real FillValue           ///< [in] value for undef entries
   );

   // officially select tracers based on configuration and allocate arrays
   static int register(
      const std::string &Name ///< [in] Name of tracer
   );

   //---------------------------------------------------------------------------
   // Query tracers
   //---------------------------------------------------------------------------

   static int getNumTracers();

   static int getIndex(
      const std::string &TracerName, ///< [in] tracer name
      int &TracerIndex               ///< [out] tracer index
   );

   static int getName(
      const int TracerIndex,  ///< [in] tracer index
      std::string &TracerName ///< [out] tracer name
   );

   // returns all tracer device arrays. If it does not exist, return nullptr
   static Array3DReal getAll(
      const int TimeIndx ///< [in] time level index
   );

   // returns a tracer device array by tracer index. If it does not exist, return nullptr
   static Array2dReal getByIndex(
      const int TracerIndex, ///< [in] global tracer index
      const int TimeIndex    ///< [in] time level index
   );

   // returns a tracer device array by tracer name. If it does not exist, return nullptr
   static Array2dReal getByName(
      const std::string &TracerName, ///< [in] global tracer name
      const int TimeIndex           ///< [in] time level index
   );

   // returns all tracer device arrays. If it does not exist, return nullptr
   static HostArray3DReal getAllHost(
      const int TimeIndx ///< [in] time level index
   );

   // returns a tracer host array by tracer index. If it does not exist, return nullptr
   static HostArray2dReal getHostByIndex(
      const int TracerIndex, ///< [in] global tracer index
      const int TimeIndex    ///< [in] time level index
   );

   // returns a tracer host array by tracer name. If it does not exist, return nullptr
   static HostArray2dReal getHostByName(
      const std::string &TracerName, ///< [in] global tracer name
      const int TimeIndex            ///< [in] time level index
   );

   // returns a field by tracer index. If it does not exist, return nullptr
   static Field getFieldByIndex(
      const int TracerIndex ///< [in] global tracer index
   );

   // returns a field by tracer name. If it does not exist, return nullptr
   static Field getFieldByName(
      const std::string &TracerName ///< [in] global tracer name
   );

   //---------------------------------------------------------------------------
   // Add a tracer to a group
   //---------------------------------------------------------------------------
   // add a tracer into a tracer group
   static int addToGroup(
      const std::string &GroupName, ///< [in] name of group
      const std::string &TracerName ///< [in] name of tracer to add
   );

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
   static void exchangeHalo(
      const int TimeLevel ///< [in] tracer time level
   );

   /// Swap time levels to update tracer arrays
   static void updateTimeLevels();

   //---------------------------------------------------------------------------
   // Device-Host data movement
   //---------------------------------------------------------------------------

   /// Copy tracers variables from host to device
   static void copyToDevice(
      const int TimeLevel ///< [in] tracer time level
   );

   /// Copy tracers variables from device to host
   static void copyToHost(
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
