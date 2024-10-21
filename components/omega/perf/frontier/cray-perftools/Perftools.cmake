# create a profile script
set(_Script ${OMEGA_BUILD_DIR}/omega_profile_perftools.sh)

file(WRITE ${_Script}  "#!/usr/bin/env bash\n\n")
file(APPEND ${_Script} "source omega_env.sh\n\n")
file(APPEND ${_Script} "module load perftools-base perftools\n\n")
file(APPEND ${_Script} "make -j ${GMAKE_J}\n\n")

foreach(_SUBLIST IN LISTS _CTESTS)

    string(REPLACE "|" ";" _ELEMENTS "${_SUBLIST}")

    list(GET _ELEMENTS 0 TEST_NAME)
    list(GET _ELEMENTS 1 EXE_NAME)
    list(GET _ELEMENTS 2 _MPI_CMD)
    string(REPLACE "_SEMICOLON_" " " MPI_CMD "${_MPI_CMD}")

    file(APPEND ${_Script} "pat_build -g hip,mpi,pnetcdf,netcdf,io -w -f test/${EXE_NAME}\n")
    file(APPEND ${_Script} "bash -c \"cd test; ${MPI_CMD} -- ../${EXE_NAME}+pat\"\n")

endforeach()

file(APPEND ${_Script} "\n\n")
file(APPEND ${_Script} "datadirs=$(find test -type d -name \"*.exe+pat+*\")\n")
file(APPEND ${_Script} "for datadir in \$datadirs; do\n")
file(APPEND ${_Script} "  if [[ -d \"$datadir/ap2-files\" || -f \"$datadir/index.ap2\" ]]; then\n")
file(APPEND ${_Script} "    rm -rf $datadir/ap2-files $datadir/index.ap2\n")
file(APPEND ${_Script} "  fi\n")
file(APPEND ${_Script} "  dirbase=$(basename \"$datadir\")\n")
file(APPEND ${_Script} "  bash -c \"cd test; pat_report ./\$dirbase > \${dirbase}_report.txt\"\n")
file(APPEND ${_Script} "done\n")

file(APPEND ${_Script} "\n\n")
file(APPEND ${_Script} "# run app2 on X-windows for visual profiling\n\n")

execute_process(COMMAND chmod +x ${_Script})
unset(_Script)
