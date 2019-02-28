#!/bin/bash
#
# Docker required!

set -eu

current=$(hg id | awk '{ print $1; }')

case "$current" in
    *+) echo "WARNING: Current working copy has been modified - build will check out the last commit, which must perforce be different";;
    *);;
esac

current=${current%%+}

rm -f vampy.so

cat Dockerfile.in | perl -p -e 's/\[\[REVISION\]\]/'"$current"'/' > Dockerfile

dockertag="cannam/vampy-$current"

sudo docker build -t "$dockertag" -f Dockerfile .

container=$(sudo docker create "$dockertag")
sudo docker cp "$container":vampy/vampy.so .
sudo docker rm "$container"

ldd vampy.so
VAMP_PATH=".:./Example VamPy plugins" ../vamp-plugin-sdk/host/vamp-simple-host -l
VAMP_PATH="." VAMPY_VERBOSE=1 ../vamp-plugin-sdk/host/vamp-simple-host -l 2>&1 | grep "Vampy version"

echo "Done!"

