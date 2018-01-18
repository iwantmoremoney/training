yes | sudo apt-get install python-pip
sudo pip install gym tensorflow keras h5py
cp -r env/.keras ~/
git submodule init
git submodule update
git config --global core.editor "vim"
cd ~ && wget -O - "https://www.dropbox.com/download?plat=lnx.x86_64" | tar xzf -
~/.dropbox-dist/dropboxd &
ln -s ../Dropbox/trained_model
echo you can try "python train.py data/NQ.csv" to verify the env
