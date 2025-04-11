#####
# Build docs
#####

# create python env:
python -m venv env_sphinx

# install requirements

pip install -U sphinx
pip install insegel

# build docs
make html
