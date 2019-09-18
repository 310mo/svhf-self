import os
import shutil
import copy

#dataからpair dataを作る（2つのディレクトリにあるjpgとnpyをもつディレクトリをたくさん作る）

#ディレクトリがなかったら作成
if not os.path.exists('pair_data'):
    os.mkdir('pair_data')

id_list = os.listdir('data')
id_list2 = copy.copy(id_list)

print(len(id_list))

count = 0
for face_id in id_list:
    for face_id2 in id_list2:
        if face_id == face_id2:
            continue
        else:
            count += 1
            
            if not os.path.exists(os.path.join('pair_data', str(count))):
                os.mkdir(os.path.join('pair_data', str(count)))
            shutil.copy(os.path.join('data', face_id, face_id+'.jpg'), os.path.join('pair_data', str(count)))
            shutil.copy(os.path.join('data', face_id, face_id+'.npy'), os.path.join('pair_data', str(count)))
            shutil.copy(os.path.join('data', face_id2, face_id2+'.jpg'), os.path.join('pair_data', str(count)))
            shutil.copy(os.path.join('data', face_id2, face_id2+'.npy'), os.path.join('pair_data', str(count)))            
            
    id_list2.remove(face_id)
