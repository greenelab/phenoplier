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
    -n|--failure-pattern)
      FAILURE_PATTERN="$2"
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
    >&2 echo "Error, --file-pattern '<value>' not provided (remember to use the single quotes)"
    exit 1
fi

if [[ -z "${SUCCESS_PATTERN}" && -z "${FAILURE_PATTERN}" ]]; then
    >&2 echo "Error, either --success-pattern <value> or --failure-pattern must be provided"
    exit 1
fi


total_count=0
not_finished_jobs=0

for logfile in $(find ${INPUT_DIR} -name "${FILE_PATTERN}"); do
    ((total_count++))

    if [ ! -z "${SUCCESS_PATTERN}" ]; then
        count=`grep -c "${SUCCESS_PATTERN}" ${logfile}`
        if [ "${count}" -lt "1" ]; then
    	    echo "WARNING, not finished yet: ${logfile}"
	    ((not_finished_jobs++))
	    continue
        fi
    else
        count=`grep -c "${FAILURE_PATTERN}" ${logfile}`
        if [ "${count}" -gt "0" ]; then
    	    echo "ERROR, failure pattern found: ${logfile}"
	    ((not_finished_jobs++))
	    continue
        fi
    fi
done

echo "Finished checking ${total_count} logs:"
if [ "${not_finished_jobs}" -eq "0" ] && [ "${total_count}" -gt 0 ]; then
    echo "  All jobs finished successfully"
elif [ "${not_finished_jobs}" -gt "0" ]; then
    echo "  ${not_finished_jobs} did not finish yet"
fi

