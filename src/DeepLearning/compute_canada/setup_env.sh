module load python/3.8 boost
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip

DIR="/mesh/"
if [ -d "$DIR" ]; then
  ### Take action if $DIR exists ###
  echo "mesh repository already available, continue with installation"
else
  ###  Control will jump here if $DIR does NOT exists ###
  echo "cloning mesh repository first, then install"
  git clone https://github.com/MPI-IS/mesh.git
fi

cd mesh
make all
cd ..
pip install -r requirements.txt
