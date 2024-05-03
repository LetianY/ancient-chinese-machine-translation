conda env remove -n "ChineseToEnglish"
conda env create -n "ChineseToEnglish" -f env_setup/Other/ChineseToEnglish.yml

## Install new environment.
python3 -m ipykernel install --user --name ChineseToEnglish --display-name "DL-S24 (3.10)"
