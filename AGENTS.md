## environment setup
first install xmake by `curl -fsSL https://xmake.io/shget.text | bash`
you need to clone and compile https://github.com/bugparty/InfiniCore.git

first make a new folder called thirdparty,go into this folder

clone https://github.com/bugparty/InfiniCore.git into InfiniCore

then install it by execute `python scripts/install.py`

after that, go back to the project root folder

execute `xmake && xmake install`