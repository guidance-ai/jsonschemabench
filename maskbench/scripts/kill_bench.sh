#!/bin/sh

kill `ps fax|grep '[r]un_maskbench' | awk '{print $1}'`
kill -9 `ps fax|grep '[r]un_maskbench' | awk '{print $1}'`
kill `ps fax|grep '[m]askbench.runner' | awk '{print $1}'`
kill -9 `ps fax|grep '[m]askbench.runner' | awk '{print $1}'`
ps fax|grep 'ma[s]kbench'
