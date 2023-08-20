#!/bin/bash
#### SETUP ####################################################################
set -o errtrace
set -o nounset

printf "\e[1m\e[7m %-80s\e[0m\n" 'isort'
isort ${@:-.}
echo

printf "\e[1m\e[7m %-80s\e[0m\n" 'black'
black ${@:-.}
echo

printf "\e[1m\e[7m %-80s\e[0m\n" 'flake8'
flake8 ${@:-.}
echo
