# create a profile script
set(_Script ${OMEGA_BUILD_DIR}/omega_profile_rocprof.sh)

file(WRITE ${_Script}  "#!/usr/bin/env bash\n\n")
file(APPEND ${_Script} "source omega_env.sh\n\n")
file(APPEND ${_Script} "make -j 8\n\n")

foreach(_SUBLIST IN LISTS _CTESTS)

    string(REPLACE "|" ";" _ELEMENTS "${_SUBLIST}")

    list(GET _ELEMENTS 0 TEST_NAME)
    list(GET _ELEMENTS 1 EXE_NAME)
    list(GET _ELEMENTS 2 _MPI_CMD)
    string(REPLACE "_SEMICOLON_" " " MPI_CMD "${_MPI_CMD}")

    file(APPEND ${_Script} "cat << 'EOF' > test/${TEST_NAME}_rocprof.sh\n")
    file(APPEND ${_Script} "#!/bin/bash\n")
    file(APPEND ${_Script} "if [[ \"$SLURM_PROCID\" -eq 0 ]]; then\n")
    file(APPEND ${_Script} "rocprof --hip-trace --hsa-trace --timestamp on -o ${TEST_NAME}.rocp ./${EXE_NAME}\n")
    file(APPEND ${_Script} "else\n")
    file(APPEND ${_Script} "./${EXE_NAME}\n")
    file(APPEND ${_Script} "fi\n")
    file(APPEND ${_Script} "EOF\n")
    file(APPEND ${_Script} "chmod +x test/${TEST_NAME}_rocprof.sh\n")

endforeach()

file(APPEND ${_Script} "for f in `ls test/*_rocprof.sh`; do\n")
file(APPEND ${_Script} "  bash -c \"cd test; ${MPI_CMD} -- $f\"\n")
file(APPEND ${_Script} "done\n")

execute_process(COMMAND chmod +x ${_Script})
unset(_Script)
