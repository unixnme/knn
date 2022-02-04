dim=$1

for batch_size in 100 300 1000; do
  for shard_size in 1000 3000 10000; do
    python main.py --backend faiss --query_file q.tsv --db_file db.tsv --dim $dim --batch_size $batch_size --shard_size $shard_size > result_faiss_${batch_size}_${shard_size}.tsv
  done
done