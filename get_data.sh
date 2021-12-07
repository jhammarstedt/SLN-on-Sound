#!/bin/bash
mkdir FSD50K
cd FSD50K
# mkdir and cd to e.g. ./datasets/fsd50k

# download, merge, uncompress and cleanup dev data
wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z01?download=1
wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z02?download=1
wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z03?download=1
wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z04?download=1
wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z05?download=1
wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.zip?download=1
for i in ./*download=1; do mv -v $i "${i/?download=1/}"; done
zip -s 0 FSD50K.dev_audio.zip --out unsplit.zip
unzip unsplit.zip  # this will uncompress the data into a folder with wav files
rm FSD50K.dev_audio.z*
rm unsplit.zip

# download, merge and uncompress eval data
wget https://zenodo.org/record/4060432/files/FSD50K.eval_audio.z01?download=1
wget https://zenodo.org/record/4060432/files/FSD50K.eval_audio.zip?download=1
for i in ./*download=1; do mv -v $i "${i/?download=1/}"; done
zip -s 0 FSD50K.eval_audio.zip --out unsplit.zip
unzip unsplit.zip  # this will uncompress the data into a folder with wav files
rm FSD50K.eval_audio.z*
rm unsplit.zip

# download and uncompress smaller files
wget https://zenodo.org/record/4060432/files/FSD50K.doc.zip?download=1
wget https://zenodo.org/record/4060432/files/FSD50K.ground_truth.zip?download=1
wget https://zenodo.org/record/4060432/files/FSD50K.metadata.zip?download=1
for i in ./*download=1; do mv -v $i "${i/?download=1/}"; done
for i in FSD50K.*.zip; do unzip $i; done  # this will uncompress the zip files into folders
for i in FSD50K.*.zip; do rm $i; done

# At this point we should be left with the filestructure as in https://zenodo.org/record/4060432
