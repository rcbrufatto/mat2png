# mat2png
Jedi Knight: Dark Forces II and MotS material extraction tool

## Installing
To install and run this tool, see below.

### Dependencies

* OpenCV 
* Numpy

To install the dependencis, do:

```
sudo apt-get install python-opencv python-numpy
```

## Directories

* cmp - *where the colormap files go.*
* palettes - *where color palettes are placed.*

## Logs

A log file is generated in the first execution and then appended after.

## Execution

Extract the files, and in the destination directory do:

```
./mat2png.py <MAT directory or MAT file> <destination directory> <colormap pathname>
```
i.e.:

```
./mat2png.py ~/mat ~/dest_mat_dir cmp/01narsh.cmp
```

Happy modding!
