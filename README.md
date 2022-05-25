# PELT
Code and data of the ACL 2022 paper "[A Simple but Effective Pluggable Entity Lookup Table for Pre-trained Language Models](https://arxiv.org/abs/2202.13392)". 


## Requiremnt
```
pip install torch
pip install numpy
pip install h5py
pip install --editable ./transformers
```
After that, install [apex](https://github.com/NVIDIA/apex) for fp16 support. 


## Generate Entity Embedding from Text
Codes are in GenerateEmbed folder.

Download linked Wikpedia from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/d5bd0071db6642699eb4/) and transform Wikipeida text to input_ids with entity position:
```
bash download_wikipedia.sh
python3 sent_filter_stp1.py
python3 sent_filter_stp2.py
```

Randomly select at most 256 sentences for each entities:
```
python3 generate_evalsamples.py
```

Obtain the output representation of masked token in entity's occurrences
```
CUDA_VISIBLE_DEVICES=0 python3 run_generate.py --model_type roberta --model_name_or_path roberta-base     --data_dir ./wiki_data  --per_gpu_eval_batch_size 256  --fp16
```

Due to limited storage space, we only upload the generated embeddings of the entities that appear in the FewRel dataset to [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/71a3262ba7614f938fb2/). If you require those of other entities, please follow the previous steps to generate new ones.



## FewRel

Codes are in the FewRel folder. Data can be found in [FewRel github repo](https://github.com/thunlp/FewRel).

Training and Evaluation:
```
CUDA_VISIBLE_DEVICES=0 python train_demo.py  --trainN  5 --N  5 --K 1 --Q 1 --model proto    --encoder robertaent --pretrain_ckpt roberta-base  --hidden_size 768  --batch_size 16 --fp16   --grad_iter  2 --train_iter 1500 --val_step 250    --cat_entity_rep  --modL 7   --seed 42  --val val_wiki  --test val_wiki
```
Add flag `--val val_pubmed --test val_pubmed` for the FewRel 2.0 dataset.


## Relation Extraction

Codes are in the RE folder. Training and Evaluation:

```
CUDA_VISIBLE_DEVICES=0 python3 run_wiki80.py     --model_type roberta  --model_name_or_path roberta-base     --do_train     --do_eval     --data_dir data/wiki80   --max_seq_length 128     --per_gpu_eval_batch_size 64       --per_gpu_train_batch_size 32 --gradient_accumulation_steps 1   --learning_rate 3e-5     --save_steps 1000  --evaluate_during_training --overwrite_output_dir   --fp16       --output_dir wiki80_models/roberta-42      --seed 42  --embed_path ../wiki_fewrel_embed/  --modL 7 --num_train_epochs 5   --train_file  train.jsonl  
```

Change `--num_train_epochs 5   --train_file  train.jsonl` to  `--num_train_epochs 50   --train_file  train_0.1.jsonl` or `--num_train_epochs 250   --train_file  train_0.01.jsonl` for the 10% and 1% training data settting.


## LAMA
Codes are in the LAMA folder. Data can be downloaded from [LAMA github repo](https://github.com/facebookresearch/LAMA).

```
Embednorm=7
python3 run_experiemnt.py 0 $Embednorm
```


## Code For BART
The code for BART is based on transformers 4.15. Please send mail to acquire this part of code. Email: ydm18@mails.tsinghua.edu.cn.

## Citation
If you use our code in your research, please cite our work:
```bibtex
@inproceedings{ye2022pelt,
  author    = {Deming Ye and
               Yankai Lin and
               Peng Li and
               Maosong Sun and
               Zhiyuan Liu},
  editor    = {Smaranda Muresan and
               Preslav Nakov and
               Aline Villavicencio},
  title     = {A Simple but Effective Pluggable Entity Lookup Table for Pre-trained
               Language Models},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational
               Linguistics (Volume 2: Short Papers), {ACL} 2022, Dublin, Ireland,
               May 22-27, 2022},
  pages     = {523--529},
  publisher = {Association for Computational Linguistics},
  year      = {2022},
  url       = {https://aclanthology.org/2022.acl-short.57},
  timestamp = {Wed, 18 May 2022 15:21:43 +0200},
  biburl    = {https://dblp.org/rec/conf/acl/YeL0S022.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
