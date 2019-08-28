echo "Compiling the library"
make
echo "Creating the library"
make create
echo "Moving header files to /usr/include"
sudo mv ../src/*.h /usr/include
echo "Moving learning lab library to /usr/lib/gcc/x86_64-linux-gnu/5"
sudo mv libllab.a /usr/lib/gcc/x86_64-linux-gnu/5

