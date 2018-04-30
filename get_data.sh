#!/bin/bash
SPLITS="train valid test"

mkdir -p "data"

##################
# PTB
##################
echo "[PTB] Downloading data..."
wget http://websail-fe.cs.northwestern.edu/downloads/cached/simple-examples.tgz
tar -xf simple-examples.tgz
mkdir "data/ptb"
for SPLIT in $SPLITS; do
  cp "simple-examples/data/ptb.$SPLIT.txt" "data/ptb/$SPLIT.txt"
done
rm simple-examples.tgz
rm -r simple-examples
echo "[PTB] Building vocab file..."
python utils/build_vocab.py data/ptb/ --end_seq

##################
# WikiText
##################
echo "[WT] Downloading data..."
wget http://websail-fe.cs.northwestern.edu/downloads/cached/wikitext.tar.gz
tar -xf wikitext.tar.gz
rm wikitext.tar.gz
CORPORA="wikitext-2 wikitext-103"
for C in $CORPORA; do
    cp -r "wikitext/"$C "data/"
done
echo "[WT] Generating vocab files..."
for C in $CORPORA; do
    python utils/build_vocab.py "data/"$C --end_seq
done

