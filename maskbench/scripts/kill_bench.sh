#!/bin/sh

kill `ps fax|grep xgr_test | awk '{print $1}'`
kill -9 `ps fax|grep xgr_test | awk '{print $1}'`
ps fax|grep xgr_test
