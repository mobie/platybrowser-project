#! /usr/bin/bash

FOLDER="install"
mkdir -p $FOLDER
cd $FOLDER

module load git
module load Java/1.8.0_221 Maven

if [ ! -d "paintera" ]; then
    git clone https://github.com/constantinpape/paintera 
fi

cd paintera
git checkout flagged-segments

mvn clean install

# copy the maven folder so we can use it for multiple users
cd ..
echo "Copying maven home"
cp -r "$HOME/.m2" .m2

cd ..
