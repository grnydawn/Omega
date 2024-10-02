//===-- ocn/TracerDefs.inc --------------------*- C++ -*-===//
//
/// \file
/// \brief OMEGA Tracer Definition File
///
/// This file defines all OMEGA tracers. It is imported using
/// the C++ #include preprocessor directive in Tracers.cpp. Only the tracer
/// definitions selected in the Tracers section of the OMEGA YAML configuration
/// file will be allocated into memory.
//===----------------------------------------------------------------------===//

// Index defined for each tracer. The value of these tracer indices are still
// set in Tracers::init()  and tracers that have not been selected would have
// standard invalid value: Tracer::IndxInvalid = -1
inline static I4 IndxTemp = -1;
inline static I4 IndxSalt = -1;
inline static I4 IndxMyBGCTracer = -1;
inline static I4 IndxAnotherBGCTracer = -1;
inline static I4 IndxDebug1 = -1;
inline static I4 IndxDebug2 = -1;

// Tracer definitions packaged in a defineAllTracers function
static void defineAllTracers() {

   IndxTemp =
       define("Temp",                  ///< [in] Name of tracer
              "Potential Temperature", ///< [in] Long name or description
              "degree_C",              ///< [in] Units
              "sea_water_potential_temperature", ///< [in] CF standard Name
              -273.15,                           ///< [in] min valid field value
              100.0,                             ///< [in] max valid field value
              1.e33,      ///< [in] value for undef entries
              &IndxTemp); ///< [out] (optional) static index

   define("Salt", "Salinity", "psu", "sea_water_salinity", 0.0, 50.0, 1.e33,
          &IndxSalt);

   define("Debug1", "Debug Tracer 1", "none", "none", 0.0, 100.0, 1.e33);
   define("Debug2", "Debug Tracer 2", "none", "none", 0.0, 100.0, 1.e33);
   define("Debug3", "Debug Tracer 3", "none", "none", 0.0, 100.0, 1.e33);
}
