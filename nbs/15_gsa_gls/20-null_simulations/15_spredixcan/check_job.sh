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
    -p|--success-pattern)
      SUCCESS_PATTERN="$2"
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

if [ -z "${SUCCESS_PATTERN}" ]; then
    >&2 echo "Error, --success-pattern <value> not provided"
    exit 1
fi


total_count=0
not_finished_jobs=0

for logfile in $(find ${INPUT_DIR} -name "*.log"); do
    ((total_count++))

    count=`grep -c "${SUCCESS_PATTERN}" ${logfile}`
    if [ "${count}" -ne "1" ]; then
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

