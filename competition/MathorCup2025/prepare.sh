# 74 s
unzip -n ./3rd_lib.zip -d ./

# DataSet 1
# Pressure
unzip -n data/data258885/train_data.zip -d ./data
unzip -n ./data/data258885/track_A.zip -d ./data/
mv ./data/track_A ./data/test_data_1
mkdir ./data/validate_data_1/
cp ./data/data/press_001.npy  ./data/test_data_1/press_658.npy
mkdir -p ./data/Test/Dataset_1/

# Dataset 2 (Building)
# geometry input: https://www.dropbox.com/scl/fo/h7q1msrlq6jdkyxxwki1s/h?dl=0&e=2
mkdir -p ./data/Training/Dataset_2/
mkdir -p ./data/Training/Dataset_2/Label_File/
cp ./data/data281827/dataset2_train_label.csv ./data/Training/Dataset_2/Label_File/
unzip -n ./data/data281827/Dataset_2_Feature.zip -d ./data/Training/Dataset_2/

mkdir -p ./data/Test/Dataset_2/Feature_File/
unzip -n ./data/data281827/test_data2_cd.zip -d ./data/Test/Dataset_2/Feature_File/
