#ifndef OMEGA_KOKKOS_H
#define OMEGA_KOKKOS_H
//===-- base/OmegaKokkos.h - Omega extension of Kokkos ------*- C++ -*-===//
//
/// \file
/// \brief Extends Kokkos for Omega uses
///
/// This header extends Kokkos for Omega uses.
//
//===----------------------------------------------------------------------===//

#include "DataTypes.h"

namespace OMEGA {

using ExecSpace = MemSpace::execution_space;

#if defined(OMEGA_ENABLE_CUDA) || defined(OMEGA_ENABLE_HIP)

using createHostMirror  = Kokkos::create_mirror_view;
using createHostCopy    = Kokkos::create_mirror_view_and_copy;

#endif

template <Int N>
using Bounds = Kokkos::MDRangePolicy<ExecSpace,
                  Kokkos::Rank<N, Kokkos::Iterate::Right, Kokkos::Iterate::Right>
               >;

// parallelFor: with label
template <Int N, class F>
inline void parallelFor(const std::string &label,
                        const Int (&upper_bounds)[N],
                        const F &f,
                        const Int (&tile)[N] = DefaultTile<N>::value) {
    if constexpr (N == 1) {
        const auto policy = Kokkos::RangePolicy(0, upper_bounds[0]);
        Kokkos::parallel_for(label, policy, f);

    } else {
        const Int lower_bounds[N] = {0};
        const auto policy = Bounds<N>(lower_bounds, upper_bounds, tile);
        Kokkos::parallel_for(label, policy, f);
  }
}

// parallelFor: without label
template <Int N, class F>
inline void parallelFor(const Int (&upper_bounds)[N],
                        const F &f,
                        const Int (&tile)[N] = DefaultTile<N>::value) {
  parallelFor("", upper_bounds, f, tile);
}

// parallelReduce: with label
template <Int N, class F, class R>
inline void parallelReduce(const std::string &label,
                           const Int (&upper_bounds)[N],
                           const F &f,
                           R &reducer,
                           const Int (&tile)[N] = DefaultTile<N>::value) {
    if constexpr (N == 1) {
        const auto policy = Kokkos::RangePolicy(0, upper_bounds[0]);
        Kokkos::parallel_reduce(label, policy, f, reducer);

    } else {
        const Int lower_bounds[N] = {0};
        const auto policy = Bounds<N>(lower_bounds, upper_bounds, tile);
        Kokkos::parallel_reduce(label, policy, f, reducer);
    }
}

// parallelReduce: without label
template <Int N, class F, class R>
inline void parallelReduce(const Int (&upper_bounds)[N],
                           const F &f,
                           R &reducer,
                           const Int (&tile)[N] = DefaultTile<N>::value) {
    parallelReduce("", upper_bounds, f, tile, reducer);
}

} // end namespace OMEGA

//===----------------------------------------------------------------------===//
#endif
