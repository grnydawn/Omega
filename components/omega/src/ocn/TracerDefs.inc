///< == arguments of define function ==
///< Name of tracer
///< Long name or description
///< Units
///< CF standard Name
///< min valid field value
///< max valid field value
///< value for undef entries

define(
   "Temp",
   "Potential Temperature",
   "degree_C",
   "sea_water_potential_temperature",
   -273.15,
   100.0,
   1.e33
);

define(
   "Salt",
   "Salinity",
   "psu",
   "sea_water_salinity",
   0.0,
   50.0,
   1.e33
);

define(
   "Debug1",
   "Debug Tracer 1",
   "none",
   "none",
   0.0,
   100.0,
   1.e33
);


define(
   "Debug2",
   "Debug Tracer 2",
   "none",
   "none",
   0.0,
   100.0,
   1.e33
);


defineR(
   "Debug3",
   "Debug Tracer 3",
   "none",
   "none",
   0.0,
   100.0,
   1.e33
);
