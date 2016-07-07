mkdir pdfoutput
rm -rf .tmp-compiling
mkdir .tmp-compiling

cd .tmp-compiling
cp ../coling2016.sty .
cp ../acl.bst .
#cp ../coling2016-relviolence.bib .
cp ../authordate1-4.sty .

mkdir ./figs
cp -r ../figs/ ./figs

pdflatex ../coling2016-relviolence.tex
# bibtex main
pdflatex ../coling2016-relviolence.tex
# pdflatex ../coling2016-relviolence.tex

cp coling2016-relviolence.pdf ../pdfoutput

cd ..
rm -rf .tmp-compiling