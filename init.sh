yes | sudo apt-get install python-pip
sudo pip install gym tensorflow keras h5py
cp -r env/.keras ~/
git submodule init
git submodule update
echo you can try "python market_dqn.py data/NQ.csv" to verify the env
