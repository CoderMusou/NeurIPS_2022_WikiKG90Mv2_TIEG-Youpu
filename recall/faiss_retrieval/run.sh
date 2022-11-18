python train_faiss_gpu.py --dstore-fp16 \
  --dimension 768 \
  --code_size 64 \
  --probe 32 \
  --dstore_mmap datastore_path.npy \
  --faiss_index directory_building_faiss_index

python faiss_search.py --topk 20000 \
  --dstore_mmap datastore_path.npy \
  --fresult validation_result2w_exclude.npy \
  --fresult_dist validation_result2w_dist_exclude.npy \
  --faiss_index directory_building_faiss_index
