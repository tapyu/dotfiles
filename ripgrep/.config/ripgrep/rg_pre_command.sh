#!/bin/sh

case $(file "$1") in
*Zstandard*)
    exec pzstd -cdq
    ;;
*)
    exec cat
    ;;
esac
