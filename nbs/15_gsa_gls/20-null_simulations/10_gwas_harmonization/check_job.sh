#!/bin/bash

# read arguments
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -i|--input-dir)
      INPUT_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    -f|--file-pattern)
      FILE_PATTERN="$2"
      shift # past argument
      shift # past value
      ;;
    -p|--success-pattern)
      SUCCESS_PATTERN="$2"
      shift # past argument
      shift # past value
      ;;
    -c|--success-pattern-count)
      SUCCESS_PATTERN_COUNT="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters


#
# check arguments
#
if [ -z "${INPUT_DIR}" ]; then
    >&2 echo "Error, --input-dir <value> not provided"
    exit 1
fi

if [ -z "${FILE_PATTERN}" ]; then
    FILE_PATTERN="*.error"
    echo "WARNING: File pattern (--file-pattern) not provided, using default: '${FILE_PATTERN}'"
fi

if [ -z "${SUCCESS_PATTERN}" ]; then
    SUCCESS_PATTERN="INFO - Finished "
    echo "WARNING: Success pattern (--success-pattern) not provided, using default: '${SUCCESS_PATTERN}'"
fi

if [ -z "${SUCCESS_PATTERN_COUNT}" ]; then
    SUCCESS_PATTERN_COUNT=1
    echo "WARNING: Success pattern count (--success-pattern-count) not provided, using default: '${SUCCESS_PATTERN_COUNT}'"
fi


total_count=0
not_finished_jobs=0

for logfile in $(find ${INPUT_DIR} -name "${FILE_PATTERN}"); do
    ((total_count++))

    count=`grep -c "${SUCCESS_PATTERN}" ${logfile}`
    if [ "${count}" -ne "${SUCCESS_PATTERN_COUNT}" ]; then
        echo "WARNING, not finished yet: ${logfile}"
        ((not_finished_jobs++))
        continue
    fi
done

echo "Finished checking ${total_count} logs:"
if [ "${not_finished_jobs}" -eq "0" ]; then
    echo "  All jobs finished successfully"
else
    echo "  ${not_finished_jobs} did not finish yet"
fi

