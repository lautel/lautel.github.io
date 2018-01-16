---
layout: post
title: Useful Linux Commands and Examples
categories: Work
---

First post! I start posting a list of handy commands when dealing with files in Linux/Unix systems. 
I work on a CentOS 6 Linux version 2.6.32

- `file -bi:` get FILE's character **encoding** (or charset).
> $ file -bi example.txt

- `iconv:` **converts** text from one encoding to another. 
> iconv -f iso-8859-1 -t utf-8 <input-text-latin1.txt >output-text-utf8.txt

- `du:` reports the amount of **disk space** used by the specified files and for each subdirectory (of directory arguments). Option [-h] needed to print sizes in human readable format. If option [--max-depth] is set to 1, it prints the total size of each folder in the directory without unfold all folders into subfolders. If it is set to 0 it displays a summary for each argument (or for the current directory). 
> $ du -h --max-depth=0 

- `wc:` **count** the number of bytes, characters, whitespace-separated words, and newlines in each given FILE, or standard input if none are given.
> $ wc example.txt &nbsp;&nbsp;&nbsp; (*same as:* $ wc -lwc example.txt)

- `cat:` **concatenate** FILEs. 
*Example:* merge all 'example' files from 1 to 5 (with a Regular Expression) into output.txt:
> $ cat example[1-5].txt > output.txt

- `grep:` searches one or more input files for lines containing a **match** to a specified pattern. Note that it doesn't modify the input file but simply prints lines matching a given pattern.

*Example:* find empty lines in a file:
> $ grep -n '^\s*$' example.txt <br /> it returns the line numbers and its content <br /><br />
> $ grep -cvP '\S' example.txt <br /> it counts the number of empty lines. It is exactly the same than <em>grep -c '^\s*$' example.txt</em> but faster, especially when searching big files.

*Example:* find which Python file (or files) within the current directory contains the word 'processing' 
> $ grep -w processing *.py <br /> argument [-w] forces to match only whole words and skip it if the target string is part of another word; e.g. 'preprocessing' will be excluded.

- `sed:` is a non-interactive stream **editor**. It is typically used to filter text, extract or subtitute multiple occurrences of a string within a file.

*Example:* replace the the word "house" by "home" every time it appears in the input file.
> $ sed 's/house/home/g' example.txt >example.out.txt

*Example:* delete the second line of the document
> $ sed -i '2d' example.txt <br /> edits the file in-place with argument [-i]. Otherwise it displays the modified file in the standard output. 

*Example:* insert a line *at the beginning* of a document:
> $ sed -i '1s/^/text-to-insert\n/' example.txt

In order to insert a line *at the end* of a document, switch to `echo` command. 
> $ echo "text-to-insert" >> example.txt

*Example:* delete all lines that contain the string 'phone' 
> $ sed -i '/phone/d' example.txt <br /> *d* at the end of the regular expression force to delete all lines matching the pattern and not only the first line.

*Example:* print lines 255 to 260 from example.txt
> $ sed -n 255,260p example.txt 

- `awk:` same as its partner *sed*, *awk* is a data file **processor**. However, it can also be a programming language itself and solve more complex tasks when writing scripts. Run a simple *awk* from the command line is also possible with a simple syntax. 

*Example:* print the third column of the file example.txt if the column separator is a comma
> $ awk -F, '{print $3}' example.txt

*Example:* save into a new file the content of example.txt adding an asterisk at the beginnig of each line
> $ cat example.txt \| awk '{print "*",$0}' > output.txt


### Resources
[GNU Operating System](https://www.gnu.org/software/) <br />
Complete command documentation running *info coreutils '[command] invocation'* 
