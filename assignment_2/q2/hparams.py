win_size = 30.0
dft_length = 1024

files = ['machali_male_8k_a.wav', 'machali_male_8k_n.wav', 'machali_male_8k_i.wav', 'machali_male_16k_s.wav']
names = {'a': '/a/', 'b': '/n/', 'c': '/I/', 'd': '/s/'}
colors = {12: 'r', 4: 'g', 6: 'b', 8: 'y', 10: 'm', 20: 'k'}
displacements = {12: 80, 4: 0, 6: 20, 8: 40, 10: 60, 20: 100}
orders = [4, 6, 8, 10, 12, 20]
file_index = 3