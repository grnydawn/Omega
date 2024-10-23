# create a profile script
set(_Script ${OMEGA_BUILD_DIR}/omega_profile_omnitrace.sh)

file(WRITE ${_Script}  "#!/usr/bin/env bash\n\n")
file(APPEND ${_Script} "source omega_env.sh\n\n")
file(APPEND ${_Script} "make -j ${GMAKE_J}\n\n")
file(APPEND ${_Script} "module load rocm omnitrace\n\n")
file(APPEND ${_Script} "omnitrace-avail -G ${OMEGA_BUILD_DIR}/omega_omnitrace.cfg --force\n")
file(APPEND ${_Script} "export OMNITRACE_CONFIG_FILE=${OMEGA_BUILD_DIR}/omega_omnitrace.cfg\n\n")


foreach(_SUBLIST IN LISTS _CTESTS)

    string(REPLACE "|" ";" _ELEMENTS "${_SUBLIST}")

    list(GET _ELEMENTS 0 TEST_NAME)
    list(GET _ELEMENTS 1 EXE_NAME)
    list(GET _ELEMENTS 2 _MPI_CMD)
    string(REPLACE "_SEMICOLON_" " " MPI_CMD "${_MPI_CMD}")

    file(APPEND ${_Script} "bash -c \"cd test; omnitrace-instrument -o ${EXE_NAME}.inst -- ./${EXE_NAME}\"\n")
    file(APPEND ${_Script} "bash -c \"cd test; ${MPI_CMD} omnitrace-run -- ./${EXE_NAME}.inst\"\n\n")
endforeach()

execute_process(COMMAND chmod +x ${_Script})
unset(_Script)
