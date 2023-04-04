# Setting up Flame in Knative

## 1 - Prerequisites
The following tools are sufficient: helm and jq. These tools can be installed by running an automation script.
- Note: We specifically install helm v3.5.1 to work with Kubernetes v1.19.0
```bash
$ cd /mydata/flame/

flame$ ./flaik/400-install-flame-prerequisites.sh
flame$ ./flaik/401-setup-cert-manager.sh
```

## (Optional) Building flame container image
To simply use flame, skip this step and go to the starting flame step. Building flame container image is only needed if a developer makes changes in the source code and wants to test the local changes.

In order to build flame container image, run the following:
```bash
$ cd /mydata/flame/flaik && sudo ./402-build-image.sh
```
To check the flame image built, run `docker images`

## 2 - Starting flame on multi-node cluster

To bring up flame and its dependent applications, `helm` is used. A shell script (`403-flame.sh`) to use helm is provided. 

### 2.1 - Configure Persistent Volume (PV)
As the Kubernetes cluster is running on top of bare metal, no dynamic provisioner is available. We need to pre-configure persistent volumes for the use of mongodb. For details please refer to [Local Persistent Volumes for Kubernetes](https://kubernetes.io/blog/2018/04/13/local-persistent-volumes-beta/), [Persistent Volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/), and [Error "no persistent volumes available for this claim and no storage class is set"](https://stackoverflow.com/questions/55780083/error-no-persistent-volumes-available-for-this-claim-and-no-storage-class-is-se)

There are two options: `NFS-PV` and `hostPath-PV`. We use `NFS-PV` by default.

---
The `NFS-PV` is enabled by specifying `mongodb.persistence.enabled` as `false` and `mongodb.persistence.storageClass` as `nfs` in `flaik/helm-chart/values.yaml`:
```yaml
mongodb:
  persistence:
    enabled: false
    storageClass: nfs
```

---
To enable `hostPath-PV`, specifying `mongodb.persistence.enabled` as `true` and `mongodb.persistence.storageClass` as `flame-storage` in `flaik/helm-chart/values.yaml`:
```yaml
mongodb:
  persistence:
    enabled: true
    storageClass: flame-storage
```
Then run the following command to create storage class and pv:
- Note: `hostPath-PV` requires **reset** for every startup of flame
```bash
$ cd /mydata/flame/flaik

# Set up persistent volume
flaik$ kubectl apply -f storage-class
flaik$ kubectl apply -f mongodb-0-pv.yaml
flaik$ kubectl apply -f mongodb-1-pv.yaml
```

### 2.2 - Configure coreDNS in Kubernetes
Add `MASTER_NOTE_IP` and `apiserver.flame.test` to /etc/hosts. See example below:
- Note: This is only required for running `flamectl`, typically on **master node**
```bash
127.0.0.1       localhost loghost localhost.sqi009-148108.kkprojects-pg0.utah.cloudlab.us
10.10.1.1       node0-link-1 node0-0 node0
10.10.1.4       node3-link-1 node3-0 node3
10.10.1.3       node2-link-1 node2-0 node2
10.10.1.5       node4-link-1 node4-0 node4
10.10.1.2       node1-link-1 node1-0 node1
128.110.218.194 flame.test apiserver.flame.test
```

Applying patch to `host{ ... }` in `coredns` ConfigMap. See example below:
- Run `kubectl edit configmap coredns -n kube-system`
- Note: This step should be automated
```json
apiVersion: v1
data:
  Corefile: |
    .:53 {
        errors
        health {
           lameduck 5s
        }
        ready
        kubernetes cluster.local in-addr.arpa ip6.arpa {
           pods insecure
           fallthrough in-addr.arpa ip6.arpa
           ttl 30
        }
        prometheus :9153
        forward . /etc/resolv.conf {
           max_concurrent 1000
        }
        cache 30
        loop
        reload
        loadbalance
        hosts {
          128.110.218.194 flame.test
          128.110.218.194 apiserver.flame.test
          128.110.218.194 notifier.flame.test
          128.110.218.194 mlflow.flame.test
          128.110.218.194 minio.flame.test
          fallthrough
        }
    }
kind: ConfigMap
metadata:
  creationTimestamp: "2023-02-23T16:29:08Z"
  name: coredns
  namespace: kube-system
  resourceVersion: "2291832"
  selfLink: /api/v1/namespaces/kube-system/configmaps/coredns
  uid: 0cf3c0c8-d799-47fe-b412-45febda56e8b
```

### 2.3 - Configure Ingress NodePort in Helm Chart
Execute `kubectl get svc ingress-nginx-controller -n ingress-nginx` to get NGINX Ingress NodePorts associated with ExternalPort *80* (HTTP) and *443* (HTTPS)

```bash
HTTPNodePort=$(kubectl -n ingress-nginx get svc ingress-nginx-controller -o jsonpath='{.spec.ports[0].nodePort}')
HTTPSNodePort=$(kubectl -n ingress-nginx get svc ingress-nginx-controller -o jsonpath='{.spec.ports[1].nodePort}')
```

Updating `HTTPNodePort` and `HTTPSNodePort` to `ingress.httpPort` and `ingress.httpsPort` in `values.yaml`:
- Note: This step should be automated
```yaml
# value.yaml
ingress:
  ...
  httpsPort: <HTTPSNodePort>
  httpPort: <HTTPNodePort>
```

### 2.4 - Starting flame control plane
```bash
$ cd /mydata/flame/flaik

flaik$ ./403-flame.sh start
```
The above command ensures that the latest official flame image from docker hub is used. To use a locally developed image, add `--local-img` in the above command.

In order to interact with flame control plane via command line, `flamectl` is required, refer to instructions at [Building flamectl](04-dev-setup.md).