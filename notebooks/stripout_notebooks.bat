call conda activate base
forfiles /m *.ipynb /c "cmd /c nbstripout @file"
