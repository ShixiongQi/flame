# Setting up Flame in Knative

## 1 - Prerequisites
The following tools are sufficient: helm and jq. These tools can be installed by running an automation script.
- Note: We specifically install helm v3.5.1 to work with Kubernetes v1.19.0
```bash
$ cd /mydata/flame/

flame$ ./flaik/400-install-flame-prerequisites.sh
flame$ ./flaik/401-setup-cert-manager.sh
```
## 2- Starting flame