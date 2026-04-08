#!/bin/bash

###################################################################
# BQ-ADMIN-SETUP.SH
# Moved from Build Folder
# Amil Khan 2022
# Updated by Wahid Sadique Koly on 2025-07-29 to align with the new upgraded codebase.
###################################################################

set -euo pipefail

VENV=${VENV:=/usr/lib/bisque}
# Rollback guard: set to 1/true/yes/on to restore legacy duplicate install pass.
BISQUE_BUILD_DOUBLE_INSTALL=${BISQUE_BUILD_DOUBLE_INSTALL:=0}
# Keep post-install config generation enabled by default.
BISQUE_BUILD_POST_FULLCONFIG=${BISQUE_BUILD_POST_FULLCONFIG:=1}

source "${VENV}/bin/activate"

# Ensure we're using Python 3
python --version

is_truthy() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

# Clean up any old .pyc files
find /source -name '*.pyc' -delete

if is_truthy "${BISQUE_BUILD_DOUBLE_INSTALL}"; then
    echo "BISQUE_BUILD_DOUBLE_INSTALL enabled: running legacy duplicate install pass"
    bq-admin setup -y install
else
    echo "Skipping duplicate install pass in bq-admin-setup.sh (already run during /builder/run-bisque.sh build)"
fi

if is_truthy "${BISQUE_BUILD_POST_FULLCONFIG}"; then
    echo "Running post-install fullconfig setup"
    bq-admin setup -y fullconfig
else
    echo "Skipping post-install fullconfig setup"
fi

# Clean up unnecessary directories
rm -rf external tools docs modules/UNPORTED

# Verify installations
echo "Verifying package installations..."
python -c "import bq.core; print('bqcore installed successfully')"
python -c "import bq.client_service; print('bqserver installed successfully')"
python -c "import bqapi; print('bqapi installed successfully')"
python -c "import bq.engine; print('bqengine installed successfully')"
python -c "import bq.features; print('bqfeature installed successfully')"

pwd
/bin/ls -l
/bin/ls -l "${VENV}/bin/"
echo "DONE"
