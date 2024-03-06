#ifndef OMEGA_DATA_TYPES_H
#define OMEGA_DATA_TYPES_H
//===-- base/DataTypes.h - data type and array definitions ------*- C++ -*-===//
//
/// \file
/// \brief Defines standard data types and Kokkos array aliases
///
/// This header defines fixed-length data types to enforce levels of precision
/// where needed. In addition, it supplies a generic real type that is double
/// precision by default but can be switched throughout using a preprocessor
/// definition SINGLE_PRECISION. Finally, all arrays in OMEGA are defined
/// as Kokkos arrays to enable allocation and kernel launching on accelerator
/// devices. Because the Kokkos definitions can be lengthy, this header defines
/// useful aliases for up to 5D arrays in all supported types on either the
/// host or device.
//
//===----------------------------------------------------------------------===//

#include "Kokkos_Core.hpp"
#include <cstdint>

namespace OMEGA {

// Standard integer and floating point types
using I4 = std::int32_t; ///< alias for 32-bit integer
using I8 = std::int64_t; ///< alias for 64-bit integer
using R4 = float;        ///< alias for 32-bit (single prec) real
using R8 = double;       ///< alias for 64-bit (double prec) real

/// generic real 64-bit (default) or 32-bit (if -DSINGLE_PRECISION used)
#ifdef SINGLE_PRECISION
using Real = float;
#else
using Real = double;
#endif

// Aliases for Kokkos arrays - by default, all arrays are on the device and
// use C-ordering.
/// Aliases for Kokkos device arrays of various dimensions and types
#ifdef KOKKOS_ENABLE_CUDA
#define DeviceMemSpace Kokkos::CudaSpace
#define DeviceLayout   Kokkos::LayoutLeft
#endif

#ifdef KOKKOS_ENABLE_HIP
#define DeviceMemSpace Kokkos::Experimental::HIPSpace
#define DeviceLayout   Kokkos::LayoutLeft
#endif

#ifdef KOKKOS_ENABLE_OPENMPTARGET
#define DeviceMemSpace Kokkos::OpenMPTargetSpace
#define DeviceLayout   Kokkos::LayoutLeft
#endif

#ifndef DeviceMemSpace
#define DeviceMemSpace Kokkos::HostSpace
#define DeviceLayout   Kokkos::LayoutRight
#endif

#define HostMemSpace Kokkos::HostSpace
#define HostLayout   Kokkos::LayoutRight

// using ExecSpace = DeviceMemSpace::execution_space;
// using range_policy = Kokkos::RangePolicy<ExecSpace>;

using Array1DI4   = Kokkos::View<I4 *, DeviceLayout, DeviceMemSpace>;
using Array1DI8   = Kokkos::View<I8 *, DeviceLayout, DeviceMemSpace>;
using Array1DR4   = Kokkos::View<R4 *, DeviceLayout, DeviceMemSpace>;
using Array1DR8   = Kokkos::View<R8 *, DeviceLayout, DeviceMemSpace>;
using Array1DReal = Kokkos::View<Real *, DeviceLayout, DeviceMemSpace>;
using Array2DI4   = Kokkos::View<I4 **, DeviceLayout, DeviceMemSpace>;
using Array2DI8   = Kokkos::View<I8 **, DeviceLayout, DeviceMemSpace>;
using Array2DR4   = Kokkos::View<R4 **, DeviceLayout, DeviceMemSpace>;
using Array2DR8   = Kokkos::View<R8 **, DeviceLayout, DeviceMemSpace>;
using Array2DReal = Kokkos::View<Real **, DeviceLayout, DeviceMemSpace>;
using Array3DI4   = Kokkos::View<I4 ***, DeviceLayout, DeviceMemSpace>;
using Array3DI8   = Kokkos::View<I8 ***, DeviceLayout, DeviceMemSpace>;
using Array3DR4   = Kokkos::View<R4 ***, DeviceLayout, DeviceMemSpace>;
using Array3DR8   = Kokkos::View<R8 ***, DeviceLayout, DeviceMemSpace>;
using Array3DReal = Kokkos::View<Real ***, DeviceLayout, DeviceMemSpace>;
using Array4DI4   = Kokkos::View<I4 ****, DeviceLayout, DeviceMemSpace>;
using Array4DI8   = Kokkos::View<I8 ****, DeviceLayout, DeviceMemSpace>;
using Array4DR4   = Kokkos::View<R4 ****, DeviceLayout, DeviceMemSpace>;
using Array4DR8   = Kokkos::View<R8 ****, DeviceLayout, DeviceMemSpace>;
using Array4DReal = Kokkos::View<Real ****, DeviceLayout, DeviceMemSpace>;
using Array5DI4   = Kokkos::View<I4 *****, DeviceLayout, DeviceMemSpace>;
using Array5DI8   = Kokkos::View<I8 *****, DeviceLayout, DeviceMemSpace>;
using Array5DR4   = Kokkos::View<R4 *****, DeviceLayout, DeviceMemSpace>;
using Array5DR8   = Kokkos::View<R8 *****, DeviceLayout, DeviceMemSpace>;
using Array5DReal = Kokkos::View<Real *****, DeviceLayout, DeviceMemSpace>;

// Also need similar aliases for arrays on the host
/// Aliases for Kokkos host arrays of various dimensions and types
using ArrayHost1DI4   = Kokkos::View<I4 *, HostLayout, HostMemSpace>;
using ArrayHost1DI8   = Kokkos::View<I8 *, HostLayout, HostMemSpace>;
using ArrayHost1DR4   = Kokkos::View<R4 *, HostLayout, HostMemSpace>;
using ArrayHost1DR8   = Kokkos::View<R8 *, HostLayout, HostMemSpace>;
using ArrayHost1DReal = Kokkos::View<Real *, HostLayout, HostMemSpace>;
using ArrayHost2DI4   = Kokkos::View<I4 **, HostLayout, HostMemSpace>;
using ArrayHost2DI8   = Kokkos::View<I8 **, HostLayout, HostMemSpace>;
using ArrayHost2DR4   = Kokkos::View<R4 **, HostLayout, HostMemSpace>;
using ArrayHost2DR8   = Kokkos::View<R8 **, HostLayout, HostMemSpace>;
using ArrayHost2DReal = Kokkos::View<Real **, HostLayout, HostMemSpace>;
using ArrayHost3DI4   = Kokkos::View<I4 ***, HostLayout, HostMemSpace>;
using ArrayHost3DI8   = Kokkos::View<I8 ***, HostLayout, HostMemSpace>;
using ArrayHost3DR4   = Kokkos::View<R4 ***, HostLayout, HostMemSpace>;
using ArrayHost3DR8   = Kokkos::View<R8 ***, HostLayout, HostMemSpace>;
using ArrayHost3DReal = Kokkos::View<Real ***, HostLayout, HostMemSpace>;
using ArrayHost4DI4   = Kokkos::View<I4 ****, HostLayout, HostMemSpace>;
using ArrayHost4DI8   = Kokkos::View<I8 ****, HostLayout, HostMemSpace>;
using ArrayHost4DR4   = Kokkos::View<R4 ****, HostLayout, HostMemSpace>;
using ArrayHost4DR8   = Kokkos::View<R8 ****, HostLayout, HostMemSpace>;
using ArrayHost4DReal = Kokkos::View<Real ****, HostLayout, HostMemSpace>;
using ArrayHost5DI4   = Kokkos::View<I4 *****, HostLayout, HostMemSpace>;
using ArrayHost5DI8   = Kokkos::View<I8 *****, HostLayout, HostMemSpace>;
using ArrayHost5DR4   = Kokkos::View<R4 *****, HostLayout, HostMemSpace>;
using ArrayHost5DR8   = Kokkos::View<R8 *****, HostLayout, HostMemSpace>;
using ArrayHost5DReal = Kokkos::View<Real *****, HostLayout, HostMemSpace>;

} // end namespace OMEGA

//===----------------------------------------------------------------------===//
#endif
