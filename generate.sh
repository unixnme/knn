set -xe

dim=$1
nq=$2
ndb=$3

python generate_vectors.py --dim $dim -n $nq --normalize > q.tsv
python generate_vectors.py --dim $dim -n $ndb --normalize --uuid > db.tsv
