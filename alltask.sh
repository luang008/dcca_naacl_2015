# Exit if any of the below listed command fails
set -e
#export PATH="/home-nfs/luang008/anaconda/bin:$PATH"

# train the deep cca network
matlab -nodesktop -nosplash -nojvm -nodisplay -r "deep_project_vectors($1,$2,$3,$4,$5,$6,$7); exit"

# apply the trained network to the word vectors
fn=result_H1=${1}_H2=${2}_rcov1=${3}_rcov2=${4}_batchsize=${5}_eta0=${6}_momentum=${7}
matlab -nodesktop -nosplash -nojvm -nodisplay -r "alltask('${fn}',$1); exit"

# paste the word to its vector
python paste.py -1 head.txt -2 ${fn}.txt -o ${fn}_inter_orig2_projected.txt -d" "
rm ${fn}.txt

# test word embeddings on 
temp=`python wordsim_simlex999.py ${fn}_inter_orig2_projected.txt`

python filterVocab.py fullVocab.txt <${fn}_inter_orig2_projected.txt>${fn}_filtVectors.txt # specific word list for ws353 task
rm ${fn}_inter_orig2_projected.txt 

temp1=`python wordsim_ws353.py ${fn}_filtVectors.txt`
rm ${fn}_filtVectors.txt
mv ${fn}.mat ./MAT
echo ${fn} ${temp} >> alltask_simlex999.txt
echo ${fn} ${temp1}  >> alltask_ws353.txt
