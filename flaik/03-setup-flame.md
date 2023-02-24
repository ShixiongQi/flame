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

## 2- Starting flame

To bring up flame and its dependent applications, `helm` is used. A shell script (`403-flame.sh`) to use helm is provided. Run the following command:
- Note: As the Kubernetes cluster is running on top of bare metal, no dynamic provisioner is available. We need to configure two persistent volumes for the use of mongodb. For details please refer to [Local Persistent Volumes for Kubernetes](https://kubernetes.io/blog/2018/04/13/local-persistent-volumes-beta/), [Persistent Volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/), and [Error "no persistent volumes available for this claim and no storage class is set"](https://stackoverflow.com/questions/55780083/error-no-persistent-volumes-available-for-this-claim-and-no-storage-class-is-se)
```bash
$ cd /mydata/flame/flaik

# Set up persistent volume
flaik$ kubectl apply -f storage-class
flaik$ kubectl apply -f mongodb-0-pv.yaml
flaik$ kubectl apply -f mongodb-1-pv.yaml

flaik$ ./403-flame.sh start
```
The above command ensures that the latest official flame image from docker hub is used. To use a locally developed image, add `--local-img` in the above command.