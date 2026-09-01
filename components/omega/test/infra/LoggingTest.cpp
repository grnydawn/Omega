//===-- Test driver for OMEGA logging -------------------------*- C++ -*-===/
//
/// \file
/// \brief Test driver for OMEGA logging
///
/// This driver tests the logging capabilities for the OMEGA
/// model. In particular, it tests creating a log file according to
/// log levels and supporting Kokkos data types.
///
//
//===-----------------------------------------------------------------------===/

#include <cstdlib>
#include <iostream>

#include "Logging.h"
#include "MachEnv.h"

#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/ringbuffer_sink.h"

#define _OMEGA_STRINGIFY(x) #x
#define _OMEGA_TOSTRING(x)  _OMEGA_STRINGIFY(x)

using namespace OMEGA;

enum CheckType { EndsWith, StartsWith, Contains };

auto TestSink = std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(1);

bool hasEnding(std::string const &fullString, std::string const &ending) {

   if (fullString.length() < ending.length()) {
      return false;
   } else {
      return (0 == fullString.compare(fullString.length() - ending.length(),
                                      ending.length(), ending));
   }
}

bool hasSubstring(std::string const &fullString, std::string const &subString) {
   return (fullString.find(subString) != std::string::npos);
}

int outputTestResult(std::string const &TestName, std::string const &Expected,
                     CheckType Type) {

   int RetVal = 0;

   std::vector<std::string> Msgs = TestSink->last_formatted();

   if (Expected.length() == 0) {
      if (Msgs.size() == 0) {
         std::cout << TestName << ": PASS" << std::endl;
      } else {
         std::cout << TestName << ": FAIL" << std::endl;
         RetVal = 1;
      }
   } else {
      if (Msgs.size() == 0) {
         std::cout << TestName << ": FAIL" << std::endl;
         RetVal = 1;
      } else {
         std::string Actual = Msgs[0];
         bool pf            = false;

         if (Type == EndsWith) {
            std::string NewExpected(Expected + "\n");
            pf = hasEnding(Actual, NewExpected);

         } else if (Type == Contains) {
            pf = hasSubstring(Actual, Expected);
         }

         if (pf) {
            std::cout << TestName << ": PASS" << std::endl;
         } else {
            std::cout << TestName << ": FAIL" << std::endl;
            RetVal = 1;
         }
      }
   }

   return RetVal;
}

int testDefaultLogLevel(bool LogEnabled) {

   int RetVal              = 0;
   bool TestRun            = true;
   const std::string TMSG1 = "This shouldn't be logged.";
   const std::string TMSG2 = "This should be logged.";

   if (OMEGA_LOG_LEVEL == 1) {
      LOG_TRACE(TMSG1);
   } else if (OMEGA_LOG_LEVEL == 2) {
      LOG_DEBUG(TMSG1);
   } else if (OMEGA_LOG_LEVEL == 3) {
      LOG_INFO(TMSG1);
   } else if (OMEGA_LOG_LEVEL == 4) {
      LOG_WARN(TMSG1);
   } else if (OMEGA_LOG_LEVEL == 5) {
      LOG_ERROR(TMSG1);
   } else {
      TestRun = false; // trace or off
   }

   if (LogEnabled && TestRun)
      RetVal +=
          outputTestResult("Default log level 1", std::string(""), EndsWith);

   TestRun = true;

   if (OMEGA_LOG_LEVEL == 0) {
      LOG_TRACE(TMSG2);
   } else if (OMEGA_LOG_LEVEL == 1) {
      LOG_DEBUG(TMSG2);
   } else if (OMEGA_LOG_LEVEL == 2) {
      LOG_INFO(TMSG2);
   } else if (OMEGA_LOG_LEVEL == 3) {
      LOG_WARN(TMSG2);
   } else if (OMEGA_LOG_LEVEL == 4) {
      LOG_ERROR(TMSG2);
   } else if (OMEGA_LOG_LEVEL == 5) {
      LOG_CRITICAL(TMSG2);
   } else {
      TestRun = false; // off
   }

   if (LogEnabled && TestRun)
      RetVal += outputTestResult("Default log level 2", TMSG2, EndsWith);

   return RetVal;
}

int testKokkosDataTypes(bool LogEnabled) {

   int RetVal       = 0;
   int constexpr d1 = 2;
   int constexpr d2 = 3;
   bool TestRun;

   Kokkos::initialize();
   {
      Array1DReal test1d("test1dD", d1);
      Array2DReal test2d("test2dD", d1, d2);

      TestRun = true;

      if (OMEGA_LOG_LEVEL == 0) {
         LOG_INFO("1d var {}", test1d);
      } else if (OMEGA_LOG_LEVEL == 1) {
         LOG_DEBUG("1d var {}", test1d);
      } else if (OMEGA_LOG_LEVEL == 2) {
         LOG_INFO("1d var {}", test1d);
      } else if (OMEGA_LOG_LEVEL == 3) {
         LOG_WARN("1d var {}", test1d);
      } else if (OMEGA_LOG_LEVEL == 4) {
         LOG_ERROR("1d var {}", test1d);
      } else if (OMEGA_LOG_LEVEL == 5) {
         LOG_CRITICAL("1d var {}", test1d);
      } else {
         TestRun = false; // off
      }

      // check if HostArray1DReal is detected
      if (LogEnabled && TestRun)
         RetVal += outputTestResult("Kokkos data type 1", "test1dD", Contains);

      TestRun = true;

      if (OMEGA_LOG_LEVEL == 0) {
         LOG_INFO("2d var {}", test2d);
      } else if (OMEGA_LOG_LEVEL == 1) {
         LOG_DEBUG("2d var {}", test2d);
      } else if (OMEGA_LOG_LEVEL == 2) {
         LOG_INFO("2d var {}", test2d);
      } else if (OMEGA_LOG_LEVEL == 3) {
         LOG_WARN("2d var {}", test2d);
      } else if (OMEGA_LOG_LEVEL == 4) {
         LOG_ERROR("2d var {}", test2d);
      } else if (OMEGA_LOG_LEVEL == 5) {
         LOG_CRITICAL("2d var {}", test2d);
      } else {
         TestRun = false; // off
      }

      // check if HostArray2DReal is detected
      if (LogEnabled && TestRun)
         RetVal += outputTestResult("Kokkos data type 2", "test2dD", Contains);

      HostArray1DReal test1dH("test1dH", d1);
      HostArray2DReal test2dH("test2dH", d1, d2);

      if (OMEGA_LOG_LEVEL == 0) {
         LOG_INFO("1d var {}", test1dH);
      } else if (OMEGA_LOG_LEVEL == 1) {
         LOG_DEBUG("1d var {}", test1dH);
      } else if (OMEGA_LOG_LEVEL == 2) {
         LOG_INFO("1d var {}", test1dH);
      } else if (OMEGA_LOG_LEVEL == 3) {
         LOG_WARN("1d var {}", test1dH);
      } else if (OMEGA_LOG_LEVEL == 4) {
         LOG_ERROR("1d var {}", test1dH);
      } else if (OMEGA_LOG_LEVEL == 5) {
         LOG_CRITICAL("1d var {}", test1dH);
      } else {
         TestRun = false; // off
      }

      // check if HostArray1DReal is detected
      if (LogEnabled && TestRun)
         RetVal += outputTestResult("Kokkos data type 1", "test1dH", Contains);

      TestRun = true;

      if (OMEGA_LOG_LEVEL == 0) {
         LOG_INFO("2d var {}", test2dH);
      } else if (OMEGA_LOG_LEVEL == 1) {
         LOG_DEBUG("2d var {}", test2dH);
      } else if (OMEGA_LOG_LEVEL == 2) {
         LOG_INFO("2d var {}", test2dH);
      } else if (OMEGA_LOG_LEVEL == 3) {
         LOG_WARN("2d var {}", test2dH);
      } else if (OMEGA_LOG_LEVEL == 4) {
         LOG_ERROR("2d var {}", test2dH);
      } else if (OMEGA_LOG_LEVEL == 5) {
         LOG_CRITICAL("2d var {}", test2dH);
      } else {
         TestRun = false; // off
      }

      // check if HostArray2DReal is detected
      if (LogEnabled && TestRun)
         RetVal += outputTestResult("Kokkos data type 2", "test2dH", Contains);
   }
   Kokkos::finalize();

   return RetVal;
}

//------------------------------------------------------------------------------
// Check the runtime flush selector parser and the resulting flush level.
//
// The parse cases are pure - no MPI, no spdlog state, no file - so they run
// and assert on every rank, unlike the checks above, which are gated on
// LogEnabled and therefore only assert on the logging rank.
int testFlushLevel(bool LogEnabled) {

   int RetVal = 0;
   bool Valid = false;

   struct FlushCase {
      const char *LevelSel;
      const char *FlushSel;
      spdlog::level::level_enum Expected;
      bool ExpectValid;
      const char *Name;
   };

   const FlushCase Cases[] = {
       // neither set: the long-standing default, unchanged
       {"", "", spdlog::level::warn, true, "default"},
       // explicit threshold, every accepted spelling
       {"trace", "", spdlog::level::trace, true, "trace"},
       {"debug", "", spdlog::level::debug, true, "debug"},
       {"info", "", spdlog::level::info, true, "info"},
       {"warn", "", spdlog::level::warn, true, "warn"},
       {"warning", "", spdlog::level::warn, true, "warning"},
       {"err", "", spdlog::level::err, true, "err"},
       {"error", "", spdlog::level::err, true, "error"},
       {"critical", "", spdlog::level::critical, true, "critical"},
       // "off" is a real level and must be reachable only when typed
       {"off", "", spdlog::level::off, true, "off"},
       // case and surrounding blanks are ignored
       {"  INFO  ", "", spdlog::level::info, true, "case and blanks"},
       // the boolean spelling of the same knob
       {"", "1", spdlog::level::info, true, "bool 1"},
       {"", "TRUE", spdlog::level::info, true, "bool true"},
       {"", "yes", spdlog::level::info, true, "bool yes"},
       {"", "on", spdlog::level::info, true, "bool on"},
       {"", "0", spdlog::level::warn, true, "bool 0"},
       {"", "off", spdlog::level::warn, true, "bool off is default"},
       // precedence: the explicit level wins when both are set
       {"critical", "1", spdlog::level::critical, true, "level beats bool"},
       // the regression that matters: a typo must NOT resolve to level::off,
       // which would never flush anything at all
       {"inof", "", spdlog::level::warn, false, "invalid level"},
       {"", "maybe", spdlog::level::warn, false, "invalid bool"},
   };

   for (const FlushCase &Case : Cases) {
      spdlog::level::level_enum Got =
          OMEGA::_selectFlushLevel(Case.LevelSel, Case.FlushSel, Valid);
      if (Got != Case.Expected || Valid != Case.ExpectValid) {
         std::cout << "FlushLevel " << Case.Name << ": FAIL" << std::endl;
         ++RetVal;
      }
   }

   // End to end: the resolved level reached the logger initLogging installed.
   // Only meaningful when neither variable is set in the environment this test
   // happens to run in, so that the expected value is the built-in default.
   const bool EnvIsClean = (std::getenv("OMEGA_LOG_FLUSH") == nullptr) &&
                           (std::getenv("OMEGA_LOG_FLUSH_LEVEL") == nullptr);
   if (LogEnabled && EnvIsClean &&
       spdlog::default_logger()->flush_level() != spdlog::level::warn) {
      std::cout << "FlushLevel applied: FAIL" << std::endl;
      ++RetVal;
   }

   if (RetVal == 0)
      std::cout << "FlushLevel: PASS" << std::endl;

   return RetVal;
}

int main(int argc, char **argv) {

   int RetVal = 0;

   MPI_Init(&argc, &argv);

   OMEGA::MachEnv::init(MPI_COMM_WORLD);
   OMEGA::MachEnv *DefEnv = OMEGA::MachEnv::getDefault();
   OMEGA::I4 TaskId       = DefEnv->getMyTask();

   try {

      std::string LogFilePath = "tmplog_" + std::to_string(TaskId) + ".log";
      std::remove(LogFilePath.c_str());

      // "sinks" hold pointers to "spdlog" sinks.
      std::vector<spdlog::sink_ptr> sinks;
      // adds the first sink of basic file sink mt
      sinks.push_back(
          std::make_shared<spdlog::sinks::basic_file_sink_mt>(LogFilePath));
      // adds the second sink for this unit testing
      sinks.push_back(TestSink);

      // creates a logger that sends log messages to multiple sinks
      auto Logger = std::make_shared<spdlog::logger>("unit", std::begin(sinks),
                                                     std::end(sinks));

      // initialize Omega logging with the logger
      bool LogEnabled = (initLogging(DefEnv, Logger) == 1);

      RetVal += testDefaultLogLevel(LogEnabled);
      RetVal += testKokkosDataTypes(LogEnabled);
      RetVal += testFlushLevel(LogEnabled);

   } catch (const std::exception &Ex) {
      std::cout << Ex.what() << ": FAIL" << std::endl;
      RetVal += 1;
   } catch (...) {
      std::cout << "Unknown: FAIL" << std::endl;
      RetVal += 1;
   }

   // Finalize environments
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Finalize();

   return RetVal;
}
