//===-- test/base/MemLeakProbe.cpp - memcheck harness self-test --*- C++ -*-===//
//
/// \file
/// \brief Deliberate-leak probe for the OMEGA_MEMCHECK harness
///
/// This program exists ONLY to prove that the configured memory-leak checker
/// actually detects leaks. It allocates and never frees a block (a "definitely
/// lost" leak). It is registered as a CTest only when OMEGA_MEMCHECK is active,
/// and it is marked WILL_FAIL: under the leak checker the definite leak makes
/// the process exit nonzero (valgrind --error-exitcode=1), so CTest sees the
/// expected failure and reports PASS. If the checker were misconfigured and did
/// NOT catch the leak, the process would exit 0 and CTest would report an
/// unexpected PASS as a FAILURE, alerting us that leak detection is broken.
//
//===----------------------------------------------------------------------===//

int main() {
   volatile int *Leak = new int[256];
   Leak[0] = 1;    // touch the allocation so it cannot be optimized away
   return 0;       // intentionally no delete[] -> definite leak
}
