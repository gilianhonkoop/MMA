# Multi Media Analytics - UvA 2025

## Snellius
### Connect to Snellius
```
ssh <user>@snellius.surf.nl 
```

### Setup project
```
cd vincent
python -m venv .venv
source .venv/bin/activate
pip install -r ../all_requirements.txt
```

This contains all the requirements for image2image and from the MMA demo provided in class.

### Access compute node
```
srun --partition=gpu_mig --gpus=1 --ntasks=1 --cpus-per-task=1 --time=00:20:00 --pty bash -i
```

### Run server on Snellius
On the root directory of the project run:
```
export PYTHONPATH="$PYTHONPATH:$PWD" 
python src/main.py
```

### Connect to server on your local machine
```
ssh -L 8050:127.0.0.1:8050 -J <user>@snellius.surf.nl <user>@<node hostname>
```

hostname will likely be: `gcn4`

After the Dash server is running open http://127.0.0.1:8050/ on your browser.


## Plotly and Dash tutorials
- Dash in 20 minutes: https://dash.plotly.com/tutorial
- Plotly plots gallery: https://plotly.com/python/

## Snellius tutorials
- Basics: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial1/Lisa_Cluster.html
- Resources: https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660209/Snellius+partitions+and+accounting