# Edit the following as needed
MY_CONDA_ENV_NAME="ChineseToEnglish"                 # name of the conda environment
MY_CONDA_ENV_CFG_FILE="env_setup/Apple M1/ChineseToEnglish-m1.yml" # name/path of the conda environment config file

main() {
    # verify conda is installed
    local conda_path="$(which conda 2>/dev/null)"
    [ -z "$conda_path" ] && { # if conda is not found, return
        echo -e $RED"conda could not be found; exiting"$NC
        return 1
    } || echo -e "conda successfully located at $conda_path"

    # verify that the yml file exists
    if [ ! -f $MY_CONDA_ENV_CFG_FILE ]; then
        echo -e $RED"The config file $MY_CONDA_ENV_CFG_FILE not found; make sure it exists in the current directory"$NC
        return 1
    fi
    # check if the env already exists
    if conda env list | command grep -q "$MY_CONDA_ENV_NAME"; then
        echo -e $CYAN"The environment $MY_CONDA_ENV_NAME may already exist; skipping creation."$NC
        return 0
    fi
    # create the actual env
    echo -e $CYAN"creating environment $MY_CONDA_ENV_NAME"$NC
    conda env create -n "$MY_CONDA_ENV_NAME" -f "$MY_CONDA_ENV_CFG_FILE" \
        && echo -e $GREEN"environment $MY_CONDA_ENV_NAME successfully created"$NC \
        && python -m ipykernel install --user --name "$MY_CONDA_ENV_NAME" --display-name "$MY_CONDA_ENV_NAME $(python -V)" \
        && echo -e $GREEN"kernel $MY_CONDA_ENV_NAME successfully installed; activating environment"$NC \
        && conda activate "$MY_CONDA_ENV_NAME" \
        && echo -e $GREEN"environment $MY_CONDA_ENV_NAME successfully activated"$NC;
}

### COLORS ####
RED='\033[1;31m'
GREEN='\033[1;32m'
CYAN='\033[1;36m'
NC='\033[0m' # No Color
###############

main "$@"
echo "exited with: $?" # return code
