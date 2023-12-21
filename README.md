# AspirationOfish
UCI - complaint Chess engine in python (almost, exept storing pv correctly...)
# Usage
1) Download python! <br>
```https://www.python.org/downloads/``` 
2) Run these commands: <br>
```
wget https://github.com/gmcheems-org/free-opening-books/raw/main/books/bin/komodo.bin
mkdir tablebase_files
cd tablebase_files
wget --mirror --no-parent --no-directories -e robots=off https://tablebase.lichess.ovh/tables/standard/3-4-5/
```
3) Install the require dependencies <br>
```
pip install -r requirements.txt
```
4) Run
```
python ofish.py
```
# Playing
Since Ofish is uci compliant chess engine, you have to use <a href="http://www.playwitharena.de/">Arena</a> or other compatible GUIS to load the engine<br>
But first, you need to build the executable by running: <br>
```
!python -m pyinstaller Ofish.py
```
Then move the folder tablebase_files and the file komodo.bin into ./dist where the executable is.


