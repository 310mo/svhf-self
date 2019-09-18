import csv
import os
import shutil

#csvファイルの読み込み
csv_file = open('vox_meta.csv', 'r')
csv_data = csv.reader(csv_file)

#データのリストを取得
idlist = os.listdir('wav')
name_list = os.listdir('unzippedFaces')

#ディレクトリがなかったら作成
if not os.path.exists('data'):
    os.mkdir('data')


for row in csv_data:
    if row[0] in idlist:
        row_id = row[0]
        row_name = row[1]
        #顔画像と音声データを入れるディレクトリを作成（なかったら）
        if not os.path.exists(os.path.join('data', row_id)):
            os.mkdir(os.path.join('data', row_id))
        target_dir = os.path.join('data', row_id)

        #音声を一つ取得
        dir_list = os.listdir(os.path.join('wav', row_id))
        file_list = os.listdir(os.path.join('wav', row_id, dir_list[0]))
        sound_file = os.path.join('wav', row_id, dir_list[0], file_list[0])
        shutil.copy(sound_file, os.path.join(target_dir, row_id+'.wav'))

        #画像を一つ取得
        image_dir_list = os.listdir(os.path.join('unzippedFaces', row_name, '1.6'))
        image_file_list = os.listdir(os.path.join('unzippedFaces', row_name, '1.6', image_dir_list[0]))
        image_file = os.path.join('unzippedFaces', row_name, '1.6', image_dir_list[0], image_file_list[0])
        shutil.copy(image_file, os.path.join(target_dir, row_id+'.jpg'))