#!/bin/bash

MODEL=transformer

for DATASET in chinese_baxter romance_ipa romance_orto
do
  # get consensus tree
  # combine all runs' trees into one file
  rm predictions/$MODEL\_$DATASET/phylogeny/consensus.newick  # in case it exists
  consense_input=predictions/$MODEL\_$DATASET/phylogeny/phylogenies
  touch $consense_input
  echo -n "" > $consense_input

  hyp_tree_list=()
  for newick in predictions/$MODEL\_$DATASET/phylogeny/*.newick
  do
    hyp_tree_list+=$newick
    cat $newick >> $consense_input
  done

  # do not want it to designate the first species as the root, so use rooted
  printf "$consense_input\nR\nY\n" | ./src/phylogeny/consense
  mv outtree predictions/$MODEL\_$DATASET/phylogeny/consensus
  rm outfile

  python src/phylogeny/fix_tree.py -t predictions/$MODEL\_$DATASET/phylogeny/consensus
  # evaluate with GQD
  hyp_tree=predictions/$MODEL\_$DATASET/phylogeny/consensus.newick
  GOLD_TREE=src/phylogeny/$DATASET\_gold.newick
  butterflies_agree=$(./src/phylogeny/quartet_dist -v $hyp_tree $GOLD_TREE | awk -F '\t' '{print $5}')
  # if you run quartet_dist on the gold tree compared to itself, # butterflies in the gold = # butterflies in the "hypothesis"
  butterflies_gold=$(./src/phylogeny/quartet_dist -v $GOLD_TREE $GOLD_TREE | awk -F '\t' '{print $5}')
  butterflies_diff=$(($butterflies_gold - $butterflies_agree))
  gqd=$(echo "scale=5; $butterflies_diff / $butterflies_gold" | bc)

  echo "$MODEL $DATASET GQD $gqd"

  # get GQD of each run's tree, then take average/stdev using numpy
  gqd_list=predictions/$MODEL\_$DATASET/phylogeny/gqd
  rm $gqd_list
  touch $gqd_list
  for hyp_tree in $hyp_tree_list
  do
    # avoid newick package's "Node names must not contain whitespace or punctuation" while reading
    sed -i.bak 's/Hong Kong/Hong_Kong/g' $hyp_tree && rm $hyp_tree.bak

    python src/phylogeny/fix_tree.py -t $hyp_tree
    mv $hyp_tree.newick $hyp_tree
    GOLD_TREE=src/phylogeny/$DATASET\_gold.newick
    butterflies_agree=$(./src/phylogeny/quartet_dist -v $hyp_tree $GOLD_TREE | awk -F '\t' '{print $5}')
    # if you run quartet_dist on the gold tree compared to itself, # butterflies in the gold = # butterflies in the "hypothesis"
    butterflies_gold=$(./src/phylogeny/quartet_dist -v $GOLD_TREE $GOLD_TREE | awk -F '\t' '{print $5}')
    butterflies_diff=$(($butterflies_gold - $butterflies_agree))
    gqd=$(echo "scale=5; $butterflies_diff / $butterflies_gold" | bc)
    echo $gqd >> $gqd_list
  done

  python -c "import pandas as pd; print( pd.read_csv('$gqd_list', header=None).mean(), pd.read_csv('$gqd_list', header=None).std() )"
done