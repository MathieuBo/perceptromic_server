import re
from os import path, mkdir
from tqdm import tqdm

def create_files(n_files):
    directory = "scripts"
    root_file = "perceptromic_root.sh"
    prefix_output_file = "{}/perceptromic_".format(directory)

    if not path.exists(directory):

        mkdir(directory)

    print("\nCreating files \n")
    for i in tqdm(range(n_files)):

        f = open(root_file, 'r')
        content = f.read()
        f.close()

        replaced = re.sub('perceptromic_job0.p 12 0', 'perceptromic_job{}.p 12 {}'.format(i, i), content)
        replaced = re.sub('Perceptromic0', 'Perceptromic{}'.format(i), replaced)

        f = open("{}{}.sh".format(prefix_output_file, i), 'w')
        f.write(replaced)
        f.close()

if __name__ == "__main__":

    create_files(n_files=1600)
