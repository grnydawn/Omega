//===-- ocn/OceanFinalize.cpp -----------------------------------*- C++ -*-===//
//
// The ocnFinalize method writes a restart file if necessary, and then cleans
// up all Omega objects
//
//===----------------------------------------------------------------------===//

#include "AuxiliaryState.h"
#include "Decomp.h"
#include "Eos.h"
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
#include "VertCoord.h"
#include "VertMix.h"

namespace OMEGA {

int ocnFinalize(const TimeInstant &CurrTime ///< [in] current sim time
) {

   // error code
   I4 RetVal = 0;

   // Write restart file if necessary

   // clean up all objects
   Tracers::clear();
   TimeStepper::clear();
   Tendencies::clear();
   AuxiliaryState::clear();
   OceanState::clear();

   // Destroy singleton instances
   Eos::destroyInstance();
   VertMix::destroyInstance();

   // Clean up I/O streams, IO system, and logging
   IOStream::finalize();
   IO::finalize();
   finalizeLogging();

   VertCoord::clear();
   Dimension::clear();
   Field::clear();
   HorzMesh::clear();
   Halo::clear();
   Decomp::clear();
   MachEnv::removeAll();

   return RetVal;
} // end ocnFinalize

} // end namespace OMEGA

//===----------------------------------------------------------------------===//
