#!/bin/sh

case "$1" in
*.pdf)
    exec pdftotext "$1" -
    ;;
*)
    case $(file "$1") in
    *Zstandard*)
        exec pzstd -cdq
        ;;
    *)
        exec cat
        ;;
    esac
    ;;
esac