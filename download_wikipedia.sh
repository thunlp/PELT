mkdir download
wget https://cloud.tsinghua.edu.cn/f/6b9ab65c68294351a408/?dl=1 -O download/0.tar.gz
wget https://cloud.tsinghua.edu.cn/f/84080b9de95d470990a4/?dl=1 -O download/1.tar.gz
wget https://cloud.tsinghua.edu.cn/f/9c6988bec0704d5ead28/?dl=1 -O download/2.tar.gz
wget https://cloud.tsinghua.edu.cn/f/3736146a599949ad9751/?dl=1 -O download/3.tar.gz
wget https://cloud.tsinghua.edu.cn/f/9a0722ac25894245a630/?dl=1 -O download/4.tar.gz
wget https://cloud.tsinghua.edu.cn/f/0408ba6ba00d4c86ad27/?dl=1 -O download/5.tar.gz
wget https://cloud.tsinghua.edu.cn/f/e3ffc06dfeb7497bbce9/?dl=1 -O download/6.tar.gz
wget https://cloud.tsinghua.edu.cn/f/65531b2b229b487fb527/?dl=1 -O download/7.tar.gz
wget https://cloud.tsinghua.edu.cn/f/d83de811729e4539b3aa/?dl=1 -O download/8.tar.gz
wget https://cloud.tsinghua.edu.cn/f/f5e8dae6a6a249c8ae1b/?dl=1 -O download/9.tar.gz
wget https://cloud.tsinghua.edu.cn/f/22cd0b37a6f1472787df/?dl=1 -O download/10.tar.gz
wget https://cloud.tsinghua.edu.cn/f/655f14f9e0114b2a9799/?dl=1 -O download/11.tar.gz
wget https://cloud.tsinghua.edu.cn/f/80283ce2fd834b248871/?dl=1 -O download/12.tar.gz
wget https://cloud.tsinghua.edu.cn/f/0412e9f1ae9140218273/?dl=1 -O download/13.tar.gz
wget https://cloud.tsinghua.edu.cn/f/022a7829b6454d6f9bd2/?dl=1 -O download/14.tar.gz
wget https://cloud.tsinghua.edu.cn/f/d304b8ee31a34c50b2ab/?dl=1 -O download/15.tar.gz
wget https://cloud.tsinghua.edu.cn/f/afc01390030e49e0b135/?dl=1 -O download/16.tar.gz
wget https://cloud.tsinghua.edu.cn/f/0ed8bb895d444f53af12/?dl=1 -O download/17.tar.gz
wget https://cloud.tsinghua.edu.cn/f/e589e25dc1cf4b0890ed/?dl=1 -O download/18.tar.gz
wget https://cloud.tsinghua.edu.cn/f/e1a4a79f6b58452fb1f5/?dl=1 -O download/19.tar.gz
wget https://cloud.tsinghua.edu.cn/f/393750f451b44abdb33a/?dl=1 -O download/20.tar.gz
wget https://cloud.tsinghua.edu.cn/f/63adfc0381034659896e/?dl=1 -O download/21.tar.gz
wget https://cloud.tsinghua.edu.cn/f/293daac5e6574029995c/?dl=1 -O download/22.tar.gz
wget https://cloud.tsinghua.edu.cn/f/8d0241e8683c4d889210/?dl=1 -O download/23.tar.gz
wget https://cloud.tsinghua.edu.cn/f/230ab0556fce415c9d49/?dl=1 -O download/24.tar.gz
wget https://cloud.tsinghua.edu.cn/f/95b6800c424f47ac873f/?dl=1 -O download/25.tar.gz
wget https://cloud.tsinghua.edu.cn/f/839ce596407f48c7922e/?dl=1 -O download/26.tar.gz
wget https://cloud.tsinghua.edu.cn/f/340b2fbb73dc402eaa7b/?dl=1 -O download/27.tar.gz
wget https://cloud.tsinghua.edu.cn/f/f41ff292f0a24d8e8dd8/?dl=1 -O download/28.tar.gz
wget https://cloud.tsinghua.edu.cn/f/54848c7aafac4e4694b9/?dl=1 -O download/29.tar.gz
wget https://cloud.tsinghua.edu.cn/f/0430f9bdd4554939bca8/?dl=1 -O download/30.tar.gz
wget https://cloud.tsinghua.edu.cn/f/d675a636a54e42baa58f/?dl=1 -O download/31.tar.gz
wget https://cloud.tsinghua.edu.cn/f/f0c6607a22b84ea28dc9/?dl=1 -O download/32.tar.gz
wget https://cloud.tsinghua.edu.cn/f/dfcf1d92badb4488b744/?dl=1 -O download/33.tar.gz
wget https://cloud.tsinghua.edu.cn/f/b86b4b63c23b467ab4e6/?dl=1 -O download/34.tar.gz
wget https://cloud.tsinghua.edu.cn/f/e5f6cff60737480db81e/?dl=1 -O download/35.tar.gz

for idx in {0..35}
do
tar xvfz download/$idx.tar.gz
done;

mv ent_quchong article_further_links_quchong

