#!/bin/bash

# if [ $# -lt 2 ]
#   then
#     echo "Provide correct number of arguments"
# 	exit 1
# fi

# design=$1
# numberOfTrainer=$2

export PATH="$HOME/.flame/bin:$PATH"

flamectl create design mnist -d "mnist_example" --insecure
flamectl create schema schema.json --design mnist --insecure
# flamectl create code $design.zip --design medmnist --insecure
flamectl create code mnist.zip --design mnist --insecure

# for i in $( eval echo {1..$numberOfTrainer} )
# do
#     flamectl create dataset dataset$i.json --insecure
# done
flamectl create dataset dataset.json --insecure

readarray -t datasetIds <<< "$(flamectl get datasets --insecure | grep MNIST | awk '{print $2}'  | awk 'NR%2==1')"
cat job.json | jq '.dataSpec.fromSystem |= []' | sponge job.json

i=0
for datasetId in "${datasetIds[@]}"
do
  cat job.json | jq --arg datasetId $datasetId --argjson i "$i" '.dataSpec.fromSystem[$i] |= . + $datasetId' | sponge job.json
  i=$((i+1))
done

flamectl create job job.json --insecure

flamectl get jobs --insecure