import os
from shutil import move


def swap_link(link, dest):
    root_folder = '/g/arendt/EM_6dpf_segmentation/EM-Prospr'
    print(link, '->', dest)

    # 1.) remove the link
    os.unlink(link)

    # 2.) move dest to link
    move(dest, link)

    # 3.) make link with relative path from dest -> link
    rel_path = os.path.relpath(os.path.abspath(link),
                               root_folder)
    # print(dest)
    # print(rel_path)
    os.symlink(rel_path, dest)


def swap_links():
    link_folder = './data/rawdata'
    cwd = os.getcwd()
    os.chdir(link_folder)
    files = os.listdir('.')

    for path in files:
        is_link = os.path.islink(path)
        if is_link:
            dest = os.readlink(path)
            assert os.path.exists(dest) and os.path.isfile(dest)
            swap_link(path, dest)

    os.chdir(cwd)


if __name__ == '__main__':
    swap_links()
