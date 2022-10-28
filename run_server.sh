# assume that the container already has activated the right environment
# asumme current working directory is at the top level of this repo
pip install --no-deps -e .
cd /home/apps/transect_viz/notebooks
panel serve transect_generated_map_animator_fabian_tract.ipynb transect_generated_map_animator.ipynb --address 0.0.0.0 --port 80 --allow-websocket-origin="*" --static-dirs transect_reports=./transect_reports
