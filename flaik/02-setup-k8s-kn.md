# Setting up Kubernetes & Knative on Cloudlab

## 1 - Download installation script to the working directory
On the master node and worker nodes, run
```bash
cd /mydata && git clone https://github.com/ShixiongQi/flame.git
cd /mydata/flame/
git checkout knative # Check to knative branch
```

## 2 - Setting up Kubernetes (v1.19.0) master node (**node-0**)
```bash
$ cd /mydata/flame/ && export MYMOUNT=/mydata

flame$ ./flaik/100-docker-install.sh && source ~/.bashrc

flame$ ./flaik/200-k8s-install.sh master 10.10.1.1

## Once the installation of Kuberentes control plane is done, 
## it will print out an token `kubeadm join ...`. 
## **PLEASE copy and save this token somewhere**. 
## The worker nodes needs this token to join the Kuberentes control plane.

flame$ echo 'source <(kubectl completion bash)' >> ~/.bashrc && source ~/.bashrc
```

## 3 - Setting up Kubernetes worker nodes (**node-1**, **node-2**, ...).
```bash
$ cd /mydata/flame/ && export MYMOUNT=/mydata

flame$ ./flaik/100-docker-install.sh && source ~/.bashrc

flame$ ./flaik/200-k8s-install.sh worker

# Use the token returned from the master node (**node-0**) to join the Kubernetes control plane. Run `sudo kubeadm join ...` with the token just saved. Please run the `kubeadm join` command with *sudo*

flame$ sudo kubeadm join <control-plane-token>
```

## 4 - Setting up Knative
On the master node (**node-0**), run
```bash
$ cd /mydata/flame
flame$ ./flaik/300-kn-install.sh
```

<!-- 1. Run `./100-docker_install.sh` without *sudo* on both *master* node and *worker* node
2. Run `source ~/.bashrc`
3. On *master* node, run `./200-k8s_install.sh master <master node IP address>`
4. On *worker* node, run `./200-k8s_install.sh worker` and then use the `kubeadm join ...` command obtained at the end of the previous step run in the master node to join the k8s cluster. Run the `kubeadm join` command with *sudo*

```
sudo kubeadm join 10.10.1.1:6443 --token btytkp.7nh8pawcdsi23g4x \
	--discovery-token-ca-cert-hash sha256:9d1802d5451e559b5c076db6901865b164bd201ed46ce38c1cba03e89618e027 
```

6. run `echo 'source <(kubectl completion bash)' >>~/.bashrc && source ~/.bashrc` -->