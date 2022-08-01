echo "Downloading data"
wget -P data/ https://docs-assets.developer.apple.com/ml-research/datasets/neuman/dataset.zip
echo "Downloding models"
wget -P out/ https://docs-assets.developer.apple.com/ml-research/datasets/neuman/pretrained.zip
echo "Extracting data"
unzip -q data/dataset.zip -d data/
echo "Extracting models"
unzip -q out/pretrained.zip -d out/
echo "Cleaning up"
mv data/dataset/* data/
mv out/pretrained/* out/
rm data/dataset.zip 
rm out/pretrained.zip
rm -r data/dataset 
rm -r data/pretrained
