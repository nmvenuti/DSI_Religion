#!/bin/bash

# Co-Occ Window Parameter
for i in {0,2,10};
do
# Context Vector Window Parameter
for j in {0..10};
do
# Start Word Freq Parameter
for k in {0,1,2,5};
do
# Eigenvector Angle Parameter 
for l in {0..4};
do

cat <<EOF > run_hyperparaop.sbatch
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --partition=serial
#SBATCH --mem=5000
#SBATCH --account=dsOne

module load anaconda/2.4
python 'masterScript.py' $i $j $k $l

EOF

#sbatch run_hyperparaop.sbatch

done
done
done
done
