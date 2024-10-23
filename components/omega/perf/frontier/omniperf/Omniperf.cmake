# create a profile script
set(_Script ${OMEGA_BUILD_DIR}/omega_profile_omniperf.sh)

file(WRITE ${_Script}  "#!/usr/bin/env bash\n\n")
file(APPEND ${_Script} "source omega_env.sh\n\n")
file(APPEND ${_Script} "make -j ${GMAKE_J}\n\n")
file(APPEND ${_Script} "module load omniperf\n\n")


foreach(_SUBLIST IN LISTS _CTESTS)

    string(REPLACE "|" ";" _ELEMENTS "${_SUBLIST}")

    list(GET _ELEMENTS 0 TEST_NAME)
    list(GET _ELEMENTS 1 EXE_NAME)
    list(GET _ELEMENTS 2 _MPI_CMD)
    string(REPLACE "_SEMICOLON_" " " MPI_CMD "${_MPI_CMD}")

    file(APPEND ${_Script} "bash -c \"cd test; ${MPI_CMD} omniperf profile -n ${TEST_NAME} --roof-only --kernel-names -- ./${EXE_NAME}\"\n")
endforeach()

execute_process(COMMAND chmod +x ${_Script})
unset(_Script)
