#!/bin/bash

TMPDIR=/home/alex/mlCourse/project/data/datasets/arabic/train

for fn in $(find . -name "*.wav"); do
  TMPFILE=$TMPDIR/$(basename $fn)
  sox $fn $TMPFILE rate 16000
  mv $TMPFILE $fn
done
