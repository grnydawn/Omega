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
#include <type_traits>

namespace OMEGA {

using ExecSpace = MemSpace::execution_space;

#ifdef OMEGA_TARGET_DEVICE

using createHostMirror  = Kokkos::create_mirror_view;
using createHostCopy    = Kokkos::create_mirror_view_and_copy;

#endif

#define deepCopy Kokkos::deep_copy

template<typename ViewType>
auto createDeviceCopy(const ViewType& sourceView, const std::string& viewName) -> Kokkos::View<typename ViewType::data_type, MemLayout, MemSpace> {
    auto destView = Kokkos::View<typename ViewType::data_type, MemLayout, MemSpace>(viewName, sourceView.layout());
    deepCopy(destView, sourceView);
    return destView;
}


/*
template<typename V>
Kokkos::View<typename V::value_type*, typename V::array_layout, typename V::memory_space> collapseArray(const typename V& src) {

    auto dst = Kokkos::View<typename V::value_type*, typename V::array_layout, typename V::memory_space>("flat" + src.label(), src.size());

    std::cout << "TTTTT " << src.rank <<  " XXXXX " << std::endl;
    for (int N=0; N<src.size(); N++) {
        if(std::is_same<typename V::array_layout, Kokkos::LayoutRight>::value) {
            if (src.rank_dynamic == 1) {
                dst(N) = src(N);
            } else if (src.rank == 2) {
                int i0 = N / src.extent(1);
                int i1 = N % src.extent(1);

                dst(N) = src(i0, i1);
            } else if (src.rank == 3) {
                int i0 = N / (src.extent(1) * src.extent(2));
                int i1 = (N / src.extent(2)) % src.extent(1);
                int i2 = N % src.extent(2);

                dst(N) = src(i0, i1, i2);
            } else if (src.Rank == 4) {
                int idx = N;
                int i0   = idx / (src.extent(1) * src.extent(2) * src.extent(3));
                idx %= (src.extent(1) * src.extent(2) * src.extent(3));
                int i1   = idx / (src.extent(2) * src.extent(3));
                idx %= (src.extent(2) * src.extent(3));
                int i2   = idx / src.extent(3);
                int i3   = idx % src.extent(3);

                dst(N) = src(i0, i1, i2, i3);
            } else if (src.Rank == 5) {
                int idx = N;
                int i0   = idx / (src.extent(1) * src.extent(2) * src.extent(3) * src.extent(4));
                idx %= (src.extent(1) * src.extent(2) * src.extent(3) * src.extent(4));
                int i1   = idx / (src.extent(2) * src.extent(3) * src.extent(4));
                idx %= (src.extent(2) * src.extent(3) * src.extent(4));
                int i2   = idx / (src.extent(3) * src.extent(4));
                idx %= (src.extent(3) * src.extent(4));
                int i3   = idx / src.extent(4);
                int i4   = idx % src.extent(4);

                dst(N) = src(i0, i1, i2, i3, i4);
            } 
        } else if(std::is_same<typename V::array_layout, Kokkos::LayoutLeft>::value) {
            if (src.Rank == 1) {
                dst(N) = src(N);
            } else if (src.Rank == 2) {
                int i0 = N % src.extent(0);
                int i1 = N / src.extent(0);

                dst(N) = src(i0, i1);
            } else if (src.Rank == 3) {
                int i0 = N % src.extent(0);
                int i1 = (N / src.extent(0)) % src.extent(1);
                int i2 = N / (src.extent(0) * src.extent(1));

                dst(N) = src(i0, i1, i2);
            } else if (src.Rank == 4) {
                int idx = N;
                int i3   = idx / (src.extent(0) * src.extent(1) * src.extent(2));
                idx %= (src.extent(0) * src.extent(1) * src.extent(2));
                int i2   = idx / (src.extent(0) * src.extent(1));
                idx %= (src.extent(0) * src.extent(1));
                int i1   = idx / src.extent(0);
                int i0   = idx % src.extent(0);

                dst(N) = src(i0, i1, i2, i3);
            } else if (src.Rank == 5) {
                int idx = N;
                int i4   = idx / (src.extent(0) * src.extent(1) * src.extent(2) * src.extent(3));
                idx %= (src.extent(0) * src.extent(1) * src.extent(2) * src.extent(3));
                int i3   = idx / (src.extent(0) * src.extent(1) * src.extent(2));
                idx %= (src.extent(0) * src.extent(1) * src.extent(2));
                int i2   = idx / (src.extent(0) * src.extent(1));
                idx %= (src.extent(0) * src.extent(1));
                int i1   = idx / src.extent(0);
                int i0   = idx % src.extent(0);

                dst(N) = src(i0, i1, i2, i3, i4);
            } 

        } else {
        // Not supported
        //cout << "collapseView support only LayoutLeft and LayoutRight\n");
        //exit(-1);
        }
    }
    return dst;
}
*/


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
