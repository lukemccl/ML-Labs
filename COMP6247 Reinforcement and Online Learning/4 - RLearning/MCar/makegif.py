import os
import imageio

for filename in os.listdir('.'):
    if('png' in filename):
        number = filename[10:].split('.')[0]
        base = 'Q_table_e_'
        rename = base + number.zfill(3) + '.png'
        os.rename(filename, rename)

with imageio.get_writer('AAAA.gif', mode='I') as writer:
        for filename in os.listdir('.'):
            if('png' in filename):
                print(filename)
                image = imageio.imread(filename)
                writer.append_data(image)