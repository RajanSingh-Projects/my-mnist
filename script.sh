#export URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl #-O tensor.whl
#pip install --upgrade $URL
wget -q $url -O image 
python test.py image 
rm image
