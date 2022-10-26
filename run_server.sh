# assume that the container already has activated the right environment
# asumme current working directory is at the top level of this repo
pip --force-reinstall Pillow
pip install --no-deps -e .
cd /home/apps/transect_viz
panel serve transect_generated_map_animator_fabian_tract.ipynb transect_generated_map_animator.ipynb --address 0.0.0.0 --port 80 --allow-websocket-origin="*" --static-dirs transect_data=./transect_data
