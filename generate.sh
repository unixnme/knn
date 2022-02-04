set -xe

dim=$1
nq=$2
ndb=$3

python generate_vectors.py --dim $dim -n $nq > q.tsv
python generate_vectors.py --dim $dim -n $ndb --uuid > db.tsv