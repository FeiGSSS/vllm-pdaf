#!/bin/bash
set -euo pipefail

scversion="stable"

if [ -d "shellcheck-${scversion}" ]; then
    export PATH="$PATH:$(pwd)/shellcheck-${scversion}"
fi

if ! [ -x "$(command -v shellcheck)" ]; then
    if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
        echo "Please install shellcheck: https://github.com/koalaman/shellcheck?tab=readme-ov-file#installing"
        exit 1
    fi

    # automatic local install if linux x86_64
    wget -qO- "https://github.com/koalaman/shellcheck/releases/download/${scversion?}/shellcheck-${scversion?}.linux.x86_64.tar.xz" | tar -xJv
    export PATH="$PATH:$(pwd)/shellcheck-${scversion}"
fi

shell_files=("$@")
if (( ${#shell_files[@]} == 0 )); then
    mapfile -d '' -t shell_files < <(
        find . -path ./.git -prune -o -name "*.sh" -print0
    )
fi

lint_status=0
for shell_file in "${shell_files[@]}"; do
    # TODO - fix warnings in .buildkite/scripts/hardware_ci/run-amd-test.sh
    if [[ "${shell_file#./}" == ".buildkite/scripts/hardware_ci/run-amd-test.sh" ]] \
        || git check-ignore -q -- "${shell_file}"; then
        continue
    fi
    if ! shellcheck -s bash -- "${shell_file}"; then
        lint_status=1
    fi
done
exit "${lint_status}"
