//===-- ocn/TendencyTerms.cpp - Tendency Terms ------------------*- C++ -*-===//
//
// The tendency terms that update state variables are implemented as functors,
// i.e. as classes that act like functions. This source defines the class
// constructors for these functors, which initialize the functor objects using
// the Mesh objects and info from the Config. The function call operators () are
// defined in the corresponding header file.
//
//===----------------------------------------------------------------------===//

#include "TendencyTerms.h"
#include "AuxiliaryState.h"
#include "Config.h"
#include "DataTypes.h"
#include "HorzMesh.h"
#include "OceanState.h"
#include "Tracers.h"

namespace OMEGA {

ThicknessFluxDivOnCell::ThicknessFluxDivOnCell(const HorzMesh *Mesh)
    : NEdgesOnCell(Mesh->NEdgesOnCell), EdgesOnCell(Mesh->EdgesOnCell),
      DvEdge(Mesh->DvEdge), AreaCell(Mesh->AreaCell),
      EdgeSignOnCell(Mesh->EdgeSignOnCell) {}

PotentialVortHAdvOnEdge::PotentialVortHAdvOnEdge(const HorzMesh *Mesh)
    : NEdgesOnEdge(Mesh->NEdgesOnEdge), EdgesOnEdge(Mesh->EdgesOnEdge),
      WeightsOnEdge(Mesh->WeightsOnEdge) {}

KEGradOnEdge::KEGradOnEdge(const HorzMesh *Mesh)
    : CellsOnEdge(Mesh->CellsOnEdge), DcEdge(Mesh->DcEdge) {}

SSHGradOnEdge::SSHGradOnEdge(const HorzMesh *Mesh)
    : CellsOnEdge(Mesh->CellsOnEdge), DcEdge(Mesh->DcEdge) {}

VelocityDiffusionOnEdge::VelocityDiffusionOnEdge(const HorzMesh *Mesh)
    : CellsOnEdge(Mesh->CellsOnEdge), VerticesOnEdge(Mesh->VerticesOnEdge),
      DcEdge(Mesh->DcEdge), DvEdge(Mesh->DvEdge),
      MeshScalingDel2(Mesh->MeshScalingDel2), EdgeMask(Mesh->EdgeMask) {}

/// F2C
VelocityDiffusionOnEdge::VelocityDiffusionOnEdge(const int NEdgesSize, const int MaxCellsOnEdge, const int NVertLevels) {

   //CellsOnEdge = MeshDecomp->CellsOnEdge; HostArray2DI4 CellsOnEdgeTmp("CellsOnEdge", NEdgesSize, MaxCellsOnEdge);
   //VerticesOnEdge = MeshDecomp->VerticesOnEdge; HostArray2DI4 VerticesOnEdgeTmp("VerticesOnEdge", NEdgesSize, 2); 
   //DcEdge = readEdgeArray(DcEdgeH, "dcEdge"); HostArray1DR8 TmpArrayR8(OmegaName + "Tmp", NEdgesSize);
   //DvEdge = readEdgeArray(DvEdgeH, "dvEdge"); HostArray1DR8 TmpArrayR8(OmegaName + "Tmp", NEdgesSize);
   //MeshScalingDel2 = Array1DReal("MeshScalingDel2", NEdgesSize);
   //EdgeMask = Array2DReal("EdgeMask", NEdgesSize, NVertLevels);

   CellsOnEdge = Array2DI4("CellsOnEdge", NEdgesSize, MaxCellsOnEdge);
   VerticesOnEdge = Array2DI4("VerticesOnEdge", NEdgesSize, 2); 
   DcEdge = Array1DReal("DcEdge", NEdgesSize);
   DvEdge = Array1DReal("DvEdge", NEdgesSize);
   MeshScalingDel2 = Array1DReal("MeshScalingDel2", NEdgesSize);
   EdgeMask = Array2DReal("EdgeMask", NEdgesSize, NVertLevels);

}

VelocityHyperDiffOnEdge::VelocityHyperDiffOnEdge(const HorzMesh *Mesh)
    : CellsOnEdge(Mesh->CellsOnEdge), VerticesOnEdge(Mesh->VerticesOnEdge),
      DcEdge(Mesh->DcEdge), DvEdge(Mesh->DvEdge),
      MeshScalingDel4(Mesh->MeshScalingDel4), EdgeMask(Mesh->EdgeMask) {}

TracerHorzAdvOnCell::TracerHorzAdvOnCell(const HorzMesh *Mesh)
    : NEdgesOnCell(Mesh->NEdgesOnCell), EdgesOnCell(Mesh->EdgesOnCell),
      CellsOnEdge(Mesh->CellsOnEdge), EdgeSignOnCell(Mesh->EdgeSignOnCell),
      DvEdge(Mesh->DvEdge), AreaCell(Mesh->AreaCell) {}

TracerDiffOnCell::TracerDiffOnCell(const HorzMesh *Mesh)
    : NEdgesOnCell(Mesh->NEdgesOnCell), EdgesOnCell(Mesh->EdgesOnCell),
      CellsOnEdge(Mesh->CellsOnEdge), EdgeSignOnCell(Mesh->EdgeSignOnCell),
      DvEdge(Mesh->DvEdge), DcEdge(Mesh->DcEdge), AreaCell(Mesh->AreaCell),
      MeshScalingDel2(Mesh->MeshScalingDel2) {}

TracerHyperDiffOnCell::TracerHyperDiffOnCell(const HorzMesh *Mesh)
    : NEdgesOnCell(Mesh->NEdgesOnCell), EdgesOnCell(Mesh->EdgesOnCell),
      CellsOnEdge(Mesh->CellsOnEdge), EdgeSignOnCell(Mesh->EdgeSignOnCell),
      DvEdge(Mesh->DvEdge), DcEdge(Mesh->DcEdge), AreaCell(Mesh->AreaCell),
      MeshScalingDel4(Mesh->MeshScalingDel4) {}

} // end namespace OMEGA

//===----------------------------------------------------------------------===//
