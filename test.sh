set -xe

dim=$1

for batch_size in 1024 2048; do
  for shard_size in 65536 1048576; do
    for backend in faiss pytorch; do
      time python knn.py --backend $backend --query_file q.tsv --db_file db.tsv --dim "$dim" --batch_size "$batch_size" --shard_size $shard_size > result_${backend}_${batch_size}_${shard_size}.tsv
    done
  done
done

file1=$(find . -name 'result_*.tsv' | head -1)
for file in $(find . -name 'result_*.tsv'); do diff "$file1" "$file"; done

rm result_*.tsv
