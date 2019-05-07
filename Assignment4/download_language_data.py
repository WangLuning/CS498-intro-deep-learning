import os
import glob
import shutil


#### Download Shakespeare file ####

os.system('wget -O shakespeare.txt https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt')


#### Download data files for classification task ####

languages = [('albanian', 'albanian'), 
             ('asv', 'english'), 
             ('czech_cep', 'czech'), 
             ('danish', 'danish'), 
             ('esperanto', 'esperanto'), 
             ('finnish_pr_1992', 'finnish'), 
             ('french_ostervald_1996', 'french'), 
             ('german_schlachter_1951', 'german'), 
             ('hungarian_karoli', 'hungarian'), 
             ('italian_riveduta_1927', 'italian'), 
             ('lithuanian', 'lithuanian'), 
             ('maori', 'maori'), 
             ('norwegian', 'norwegian'), 
             ('portuguese', 'portuguese'), 
             ('romanian_cornilescu', 'romanian'), 
             ('spanish_reina_valera_1909', 'spanish'), 
             ('swedish_1917', 'swedish'), 
             ('turkish', 'turkish'), 
             ('vietnamese_1934', 'vietnamese'), 
             ('xhosa', 'xhosa')]

os.mkdir('./language_data')

data_dir_base = './language_data/'

for (download_name, dest_name) in languages:
    # download
    path_base = 'http://unbound.biola.edu/downloads/bibles/' + download_name + '.zip'
    print('downloading from: ', path_base)
    os.system('wget '+ path_base)
    
    # unzip 
    unzip_dir = data_dir_base+download_name+'_unzipped'
    print('unzipping to: ', unzip_dir)
    os.makedirs(unzip_dir)
    os.system('unzip '+ (download_name + '.zip') + ' -d ' + unzip_dir)
    
    # move text file to new file name
    text_file_path = unzip_dir+'/'+download_name+'_utf8.txt'
    new_text_file_path = data_dir_base+dest_name+'.txt'
    print('keeping file: ', new_text_file_path)
    os.system('mv' + ' ' + text_file_path + ' ' + new_text_file_path)
    
    # remove zip file and inflated 
    print('cleaning up directory')
    os.remove(download_name+'.zip')
    shutil.rmtree(unzip_dir)


### Split language files into train and test ###

files_list = glob.glob('language_data/*.txt')

if not os.path.exists('language_data/test'):
    os.makedirs('language_data/test')
if not os.path.exists('language_data/train'):
    os.makedirs('language_data/train')

for file in files_list:
    bible_train_text = ""
    bible_test_text = ""
  
    new_train_file_name = 'language_data/train/'+os.path.basename(file).split(".")[0] + "_train.txt"
    new_test_file_name = 'language_data/test/'+os.path.basename(file).split(".")[0] + "_test.txt"
    with open(file) as inp:
        for _ in range(8):
            next(inp)

        # Create training set text file
        for i in range(8,27995):
            line = next(inp)
            tab_sep = line.strip().replace('\t\t', '\t').split('\t')
            words = tab_sep[-1]
            bible_train_text += " " + words

        with open(new_train_file_name, 'w') as f:
            f.write(bible_train_text.strip())

        # Create testing set text file
        for test_lines in inp:
            tab_sep_test = test_lines.strip().replace('\t\t', '\t').split('\t')
            words_test = tab_sep_test[-1]
            bible_test_text += " " + words_test

        with open(new_test_file_name, 'w') as f:
            f.write(bible_test_text.strip())
    
    # Remove old combined text file
    os.remove(file)


#### Download test language file for Kaggle submission #### 

os.system('wget -O language_data/kaggle_rnn_language_classification_test.txt https://uofi.box.com/shared/static/094rb0n0serfb1s19iwrvk0vwhf8bg6p.txt')

