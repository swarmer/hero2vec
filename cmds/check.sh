#!/usr/bin/env bash

# coloring
bold=$(tput bold)
green=$(tput setaf 2)
normal=$(tput sgr0)

echo "${bold}mypy${normal}"
mypy --ignore-missing-imports hero2vec \
    && echo "${green}OK${normal}" \
    || exit 1
echo

echo "${bold}pylint${normal}"
pylint hero2vec \
    && echo -e "${green}OK${normal}\n" \
    || exit 1

echo "${green}All OK${normal}"
