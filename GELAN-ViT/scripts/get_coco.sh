#!/bin/bash
# COCO 2017 dataset http://cocodataset.org
# Download command: bash ./scripts/get_coco.sh

# Download/unzip images
d='./coco/images' # unzip directory
url=http://images.cocodataset.org/zips/
f2='val2017.zip'   # 1G, 5k images
for f in $f2 ; do
  echo 'Downloading' $url$f '...'
  unzip -q $f -d $d && rm $f & # download, unzip, remove in background
done
wait # finish background tasks
