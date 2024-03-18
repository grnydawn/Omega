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
//#include <cstdint>

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

using Int  = int;

// user-defined literal for generic reals
KOKKOS_INLINE_FUNCTION constexpr Real operator""_Real(long double x) {
   return x;
}

// Aliases for Kokkos arrays - by default, all arrays are on the device and
// use C-ordering.
/// Aliases for Kokkos device arrays of various dimensions and types

#ifdef OMEGA_ENABLE_CUDA
using MemSpace  = Kokkos::CudaSpace;
using MemLayout = Kokkos::LayoutRight;
#endif

#ifdef OMEGA_ENABLE_HIP
using MemSpace  = Kokkos::Experimental::HIPSpace;
using MemLayout = Kokkos::LayoutRight;
#endif

#ifdef OMEGA_ENABLE_OPENMP
using MemSpace  = Kokkos::HostSpace;
using MemLayout = Kokkos::LayoutRight;
#endif

using HostMemSpace  = Kokkos::HostSpace;
using HostMemLayout = Kokkos::LayoutRight;

template <Int N> struct DefaultTile;

template <> struct DefaultTile<1> {
    static constexpr Int value[] = {64};
};

template <> struct DefaultTile<2> {
    static constexpr Int value[] = {1, 64};
};

template <> struct DefaultTile<3> {
    static constexpr Int value[] = {1, 1, 64};
};

template <> struct DefaultTile<4> {
    static constexpr Int value[] = {1, 1, 1, 64};
};

template <> struct DefaultTile<5> {
    static constexpr Int value[] = {1, 1, 1, 1, 64};
};

using Array1DI4   = Kokkos::View<I4 *, MemLayout, MemSpace>;
using Array1DI8   = Kokkos::View<I8 *, MemLayout, MemSpace>;
using Array1DR4   = Kokkos::View<R4 *, MemLayout, MemSpace>;
using Array1DR8   = Kokkos::View<R8 *, MemLayout, MemSpace>;
using Array1DReal = Kokkos::View<Real *, MemLayout, MemSpace>;
using Array2DI4   = Kokkos::View<I4 **, MemLayout, MemSpace>;
using Array2DI8   = Kokkos::View<I8 **, MemLayout, MemSpace>;
using Array2DR4   = Kokkos::View<R4 **, MemLayout, MemSpace>;
using Array2DR8   = Kokkos::View<R8 **, MemLayout, MemSpace>;
using Array2DReal = Kokkos::View<Real **, MemLayout, MemSpace>;
using Array3DI4   = Kokkos::View<I4 ***, MemLayout, MemSpace>;
using Array3DI8   = Kokkos::View<I8 ***, MemLayout, MemSpace>;
using Array3DR4   = Kokkos::View<R4 ***, MemLayout, MemSpace>;
using Array3DR8   = Kokkos::View<R8 ***, MemLayout, MemSpace>;
using Array3DReal = Kokkos::View<Real ***, MemLayout, MemSpace>;
using Array4DI4   = Kokkos::View<I4 ****, MemLayout, MemSpace>;
using Array4DI8   = Kokkos::View<I8 ****, MemLayout, MemSpace>;
using Array4DR4   = Kokkos::View<R4 ****, MemLayout, MemSpace>;
using Array4DR8   = Kokkos::View<R8 ****, MemLayout, MemSpace>;
using Array4DReal = Kokkos::View<Real ****, MemLayout, MemSpace>;
using Array5DI4   = Kokkos::View<I4 *****, MemLayout, MemSpace>;
using Array5DI8   = Kokkos::View<I8 *****, MemLayout, MemSpace>;
using Array5DR4   = Kokkos::View<R4 *****, MemLayout, MemSpace>;
using Array5DR8   = Kokkos::View<R8 *****, MemLayout, MemSpace>;
using Array5DReal = Kokkos::View<Real *****, MemLayout, MemSpace>;

// Also need similar aliases for arrays on the host
/// Aliases for Kokkos host arrays of various dimensions and types
using HostArray1DI4   = Kokkos::View<I4 *, HostMemLayout, HostMemSpace>;
using HostArray1DI8   = Kokkos::View<I8 *, HostMemLayout, HostMemSpace>;
using HostArray1DR4   = Kokkos::View<R4 *, HostMemLayout, HostMemSpace>;
using HostArray1DR8   = Kokkos::View<R8 *, HostMemLayout, HostMemSpace>;
using HostArray1DReal = Kokkos::View<Real *, HostMemLayout, HostMemSpace>;
using HostArray2DI4   = Kokkos::View<I4 **, HostMemLayout, HostMemSpace>;
using HostArray2DI8   = Kokkos::View<I8 **, HostMemLayout, HostMemSpace>;
using HostArray2DR4   = Kokkos::View<R4 **, HostMemLayout, HostMemSpace>;
using HostArray2DR8   = Kokkos::View<R8 **, HostMemLayout, HostMemSpace>;
using HostArray2DReal = Kokkos::View<Real **, HostMemLayout, HostMemSpace>;
using HostArray3DI4   = Kokkos::View<I4 ***, HostMemLayout, HostMemSpace>;
using HostArray3DI8   = Kokkos::View<I8 ***, HostMemLayout, HostMemSpace>;
using HostArray3DR4   = Kokkos::View<R4 ***, HostMemLayout, HostMemSpace>;
using HostArray3DR8   = Kokkos::View<R8 ***, HostMemLayout, HostMemSpace>;
using HostArray3DReal = Kokkos::View<Real ***, HostMemLayout, HostMemSpace>;
using HostArray4DI4   = Kokkos::View<I4 ****, HostMemLayout, HostMemSpace>;
using HostArray4DI8   = Kokkos::View<I8 ****, HostMemLayout, HostMemSpace>;
using HostArray4DR4   = Kokkos::View<R4 ****, HostMemLayout, HostMemSpace>;
using HostArray4DR8   = Kokkos::View<R8 ****, HostMemLayout, HostMemSpace>;
using HostArray4DReal = Kokkos::View<Real ****, HostMemLayout, HostMemSpace>;
using HostArray5DI4   = Kokkos::View<I4 *****, HostMemLayout, HostMemSpace>;
using HostArray5DI8   = Kokkos::View<I8 *****, HostMemLayout, HostMemSpace>;
using HostArray5DR4   = Kokkos::View<R4 *****, HostMemLayout, HostMemSpace>;
using HostArray5DR8   = Kokkos::View<R8 *****, HostMemLayout, HostMemSpace>;
using HostArray5DReal = Kokkos::View<Real *****, HostMemLayout, HostMemSpace>;

} // end namespace OMEGA

//===----------------------------------------------------------------------===//
#endif
