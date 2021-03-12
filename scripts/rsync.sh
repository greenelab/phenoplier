#!/bin/bash

rsync \
        -chavzP \
        --stats \
        --exclude 'recount_data_prep_PLIER.*' \
        --exclude 'recount2_PLIER_data.zip' \
        --exclude 'recount_PLIER_model.RDS' \
        --exclude 'srp/' \
        $@

