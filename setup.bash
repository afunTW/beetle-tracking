#!/bin/bash

GITHUB_DETECTION="https://github.com/afunTW/TensorBox"
GITHUB_CLASSIFICATION="https://github.com/afunTW/keras-transfer-workflow"

# check the env for pyenv if the system do not have the right version of Pipfile
if [ ! -d "$HOME/.pyenv" ]; then
	sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
	libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
	xz-utils tk-dev libffi-dev

	git clone https://github.com/pyenv/pyenv.git ~/.pyenv
	echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
	echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
	echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.profile
	source ~/.profile
fi

# check detection and classfication site-project
if [ ! -d "detection" ]; then
	echo "Can't find the detection folder, clone from github..."
	git clone $GITHUB_DETECTION detection
	cd detection && pipenv sync
	source $(pipenv --venv)/bin/activate
	cd utils && make && deactivate && cd ../..
fi
if [ ! -d "classification" ]; then
	echo "Can't find the classification folder, clone from github..."
	git clone $GITHUB_CLASSIFICATION classification
	cd classification && pipenv sync && cd ..
fi