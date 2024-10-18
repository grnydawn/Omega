# create a profile script
set(_Script ${OMEGA_BUILD_DIR}/omega_profile_perftools.sh)

file(WRITE ${_Script}  "#!/usr/bin/env bash\n\n")
file(APPEND ${_Script} "source omega_env.sh\n\n")
file(APPEND ${_Script} "module load perftools-base perftools\n\n")
file(APPEND ${_Script} "make -j 8\n\n")

foreach(_SUBLIST IN LISTS _CTESTS)

    string(REPLACE "|" ";" _ELEMENTS "${_SUBLIST}")

    list(GET _ELEMENTS 0 TEST_NAME)
    list(GET _ELEMENTS 1 EXE_NAME)
    list(GET _ELEMENTS 2 _MPI_CMD)
    string(REPLACE "_SEMICOLON_" " " MPI_CMD "${_MPI_CMD}")

    file(APPEND ${_Script} "pat_build -g hip,mpi,pnetcdf,netcdf,io -w -f test/${EXE_NAME}\n")
    file(APPEND ${_Script} "bash -c \"cd test; ${MPI_CMD} -- ../${EXE_NAME}+pat\"\n")

endforeach()

file(APPEND ${_Script} "for f in `ls test/*.exe+pat+*`; do\n")
file(APPEND ${_Script} "    pat_report $f > $f_report.txt\n")
file(APPEND ${_Script} "done\n")

file(APPEND ${_Script} "# run app2 on X-windows for visual profiling\n\n")

execute_process(COMMAND chmod +x ${_Script})
unset(_Script)
