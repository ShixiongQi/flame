#!/usr/bin/env bash
# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

function install_ubuntu_prerequisites {
    install_helm

    install_jq
}

function install_helm {
    echo "Installing helm"
    if ! command -v helm &> /dev/null
    then
        echo "Downloading helm installation script"
        curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
        chmod 700 get_helm.sh
        DESIRED_VERSION=v3.5.1 ./get_helm.sh
        echo "helm installed"
        echo "Deleting helm installation script"
        rm get_helm.sh
    else
        echo "helm already installed. Skipping..."
    fi
}

function install_jq {
    echo "Installing jq"
    if ! command -v jq &> /dev/null
    then
        wget -O jq https://github.com/stedolan/jq/releases/download/jq-1.6/jq-linux64
        chmod +x ./jq
        sudo mv jq /usr/bin
        echo "jq installed"
    else
        echo "jq already installed. Skipping..."
    fi
}

install_ubuntu_prerequisites