# Development environment Setup

## Prerequisites

The target development environment is Ubuntu 20.04. This section describes how to set up a development environment in Ubuntu 20.04.

The following tools and packages are needed as minimum:
- go 1.16+
- golangci-lint
- python 3.9+

---
The following shows how to install the above packages in Ubuntu 20.04.

First, keep package list and their dependencies up to date.
```bash
sudo apt update
```

Install golang and and golangci-lint.
```bash
golang_file=go1.18.6.linux-amd64.tar.gz
curl -LO https://go.dev/dl/$golang_file && tar -C $HOME -xzf $golang_file
echo "PATH=\"\$HOME/go/bin:\$PATH\"" >> $HOME/.bashrc
source $HOME/.bashrc

curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(go env GOPATH)/bin v1.49.0
golangci-lint --version
```

Then install pyenv with the following commands:
```bash
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev

curl https://pyenv.run | bash

echo "" >> $HOME/.bashrc
echo "PATH=\"\$HOME/.pyenv/bin:\$PATH\"" >> $HOME/.bashrc
echo "eval \"\$(pyenv init --path)\"" >> $HOME/.bashrc
echo "eval \"\$(pyenv virtualenv-init -)\"" >> $HOME/.bashrc
source $HOME/.bashrc
```

Using `pyenv`, install python version 3.9.6.
```bash
pyenv install 3.9.6
pyenv global 3.9.6
```
To check the version, run `pyenv version` and `python --version`, an example output looks like the following:
```bash
vagrant@flame:~$ pyenv version
3.9.6 (set by /home/vagrant/.pyenv/version)
vagrant@flame:~$ python --version
Python 3.9.6
```

## Creating flame config
The following command creates `config.yaml` under `$HOME/.flame`.
```bash
./build-config.sh
```

## Building flamectl
The flame CLI tool, `flamectl` uses the configuration file (`config.yaml`) to interact with the flame system.
In order to build `flamectl`, run `make install` from the level folder (i.e., `flame`).
This command compiles source code and installs `flamectl` binary as well as other binaries into `$HOME/.flame/bin`.
You may want to add `export PATH="$HOME/.flame/bin:$PATH"` to your shell config (e.g., `~/.zshrc`, `~/.bashrc`) and then reload your shell config (e.g., `source ~/.bashrc`).