set -xe

dim=$1

for backend in faiss pytorch; do
  for batch_size in 100 300 1000; do
    for shard_size in 1000 3000 10000; do
      time python main.py --backend $backend --query_file q.tsv --db_file db.tsv --dim "$dim" --batch_size "$batch_size" --shard_size $shard_size > result_${backend}_${batch_size}_${shard_size}.tsv
    done
  done
done

for file in $(find . -name 'result_*.tsv'); do diff result_faiss_1000_1000.tsv "$file"; done

rm result_*.tsv