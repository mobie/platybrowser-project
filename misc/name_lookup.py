import os
import json

# From Detlev's mail
NEW_GENE_NAMES = {
    "NOV1": "globin-like",
    "NOV2": "NOV2",
    "NOV6": "Stathmin",
    "NOV15": "OLM2A",
    "NOV18": "CEPU1",
    "NOV29": "NOV29",
    "NOV45": "TMTC3",
    "NOV50": "GDPD1",
    "NOV52": "KANL3",
    "ENR1": "PGAM",
    "ENR2": "RYR2",
    "ENR3": "JKIP3",
    "ENR4": "SND1",
    "ENR6": "Nucleolin",
    "ENR8": "Non-muscle-MHC",
    "ENR9": "NOE1",
    "ENR10": "UPP",
    "ENR12": "UNC22",
    "ENR13": "NDUS1",
    "ENR16": "ODO2",
    "ENR19": "FXL21",
    "ENR20": "PPIB",
    "ENR22": "CO1A1",
    "ENR25": "Synaptopodin",
    "ENR29": "USP9X",
    "ENR30": "CCVD",
    "ENR31": "Leucin-rich",
    "ENR32": "GRIK3",
    "ENR34": "MTHFSD",
    "ENR39": "RPC2",
    "ENR46": "Calexcitin2",
    "ENR54": "Boule-like",
    "ENR57": "Junctophilin1",
    "ENR62": "NB5R3",
    "ENR64": "PSMF1",
    "ENR69": "BCA1",
    "ENR71": "Patched"
}

#
DYNAMIC_SEGMENTATIONS = ['sbem-6dpf-1-whole-segmented-cells',
                         'sbem-6dpf-1-whole-segmented-cilia',
                         'sbem-6dpf-1-whole-segmented-nuclei']

ROOT = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data'
LUT_PATH = 'new_name_lut.json'

FILE_NAME_LUT = {}
IMAGE_PROPERTIES = {}


# we need to make the following update to names:
# - prospr -> gene names need to be replaced according to list
#             (including to two region names !) and be lowercase
# - prospr -> get rid of the '-MED' postfix
# - segmentations -> get rid of '-labels' postifx
def update_name_lut():
    global FILE_NAME_LUT
    if os.path.exists(LUT_PATH):
        with open(LUT_PATH, 'r') as f:
            FILE_NAME_LUT.update(json.load(f))
        return

    # update files according to the last version folder
    folder = os.path.join(ROOT, '0.6.5')
    image_names = os.listdir(os.path.join(folder, 'images'))
    image_names = [os.path.splitext(name)[0] for name in image_names
                   if os.path.splitext(name)[1] == '.xml']
    seg_names = os.listdir(os.path.join(folder, 'segmentations'))
    seg_names = [os.path.splitext(name)[0] for name in seg_names
                 if os.path.splitext(name)[1] == '.xml']

    file_names = image_names + seg_names
    for name in file_names:
        new_name = name

        # get rid of '-MED'
        if '-MED' in new_name:
            new_name = new_name.replace('-MED', '')
        # get rid of '-labels'
        if '-labels' in new_name:
            new_name = new_name.replace('-labels', '')
        # get rid of '-ariande' tag [sic]
        if '-ariande' in new_name:
            new_name = new_name.replace('-ariande', '')
        # replace '-mask' with '-segmented'
        if '-mask' in new_name:
            new_name = new_name.replace('-mask', '-segmented')

        # update the gene / region names for prospr
        # and make everything lowercase
        if new_name.startswith('prospr'):

            # replace gene names / region names
            gene_name = new_name.split('-')[4:]
            if gene_name[0] in NEW_GENE_NAMES:
                gene_name = [NEW_GENE_NAMES[gene_name[0]]]
            elif len(gene_name) > 1 and gene_name[1] == 'PNS':
                gene_name = ['segmented', 'lateralectoderm']
            elif len(gene_name) > 1 and gene_name[1] == 'Stomodeum':
                gene_name = ['segmented', 'foregut']
            new_name = '-'.join(new_name.split('-')[:4] + gene_name)

            # make lowercase
            new_name = new_name.lower()

        FILE_NAME_LUT[name] = new_name


def update_image_properties():
    global IMAGE_PROPERTIES
    for name in FILE_NAME_LUT.values():
        properties = {'Storage': {'local': 'local/%s.xml' % name}}
        table_folder = 'tables/%s' % name

        # prospr: Color Magenta
        #         value range 0 - 1000
        if name.startswith('prospr'):
            if 'virtual-cells' in name:
                properties.update({'ColorMap': 'Glasbey', 'TableFolder': table_folder})
            else:
                properties.update({'Color': 'Magenta', 'MinValue': 0, 'MaxValue': 1000})

        # handle all real segmentations with glasbey color map and tables
        # - cells
        # - chromatin
        # - cilia
        # - ganglia
        # - nuclei
        # - tissue
        elif ('segmented-cells' in name
              or 'segmented-chromatin' in name
              or 'segmented-cilia' in name
              or 'segmented-ganglia' in name
              or 'segmented-nuclei' in name
              or 'segmented-tissue' in name):
            properties.update({'ColorMap': 'Glasbey', 'TableFolder': table_folder})

        # all other segmentations are binary masks
        elif '-segmented' in name:
            properties.update({'Color': 'White', 'MinValue': 0, 'MaxValue': 1})

        # em-raw: Color White
        #         value range 0 - 255
        else:
            properties.update({'Color': 'White', 'MinValue': 0, 'MaxValue': 255})

        IMAGE_PROPERTIES[name] = properties


update_name_lut()
update_image_properties()


def look_up_filename(file_name):
    new_file_name = FILE_NAME_LUT.get(file_name, None)
    # Try to match ENR/NOV filenames
    if new_file_name is None:
        old_gene_name = file_name.split('-')[4]
        # hox5 was renamed to hox4
        if old_gene_name.lower() == 'hox5':
            gene_name = 'hox4'
        # irx was renamed to irx6
        elif old_gene_name.lower() == 'irx':
            gene_name = 'irx6'
        # prospr reference volume was renamed
        elif old_gene_name.lower() == 'ref':
            gene_name = 'segmented-prospr6-ref'
        # muscles lost an s at some point
        elif old_gene_name == 'segmented' and file_name.split('-')[5] == 'muscles':
            gene_name = 'segmented-muscle'
        else:
            assert old_gene_name in NEW_GENE_NAMES, file_name
            gene_name = NEW_GENE_NAMES[old_gene_name].lower()
        new_file_name = '-'.join(file_name.split('-')[:4] + [gene_name])
        assert new_file_name in FILE_NAME_LUT.values(), new_file_name
    return new_file_name


def get_image_properties(name):
    return IMAGE_PROPERTIES[name]


# TODO currently we have a lot of different version of paintera projects.
# for cells and cilia, the most up-to-date are actually the label-multiset variants
# need to clean that up and move the most up to date versions to the names used here,
# but need to coordinate with valentyna first
def get_dynamic_segmentation_properties(name):
    # cell segmentation
    if name == DYNAMIC_SEGMENTATIONS[0]:
        return {'PainteraProject': ['/g/kreshuk/data/arendt/platyneris_v1/data.n5',
                                    'volumes/paintera/proofread_cells'],
                'TableUpdateFunction': 'make_cell_tables',
                'Postprocess': {"BoundaryPath": "/g/kreshuk/data/arendt/platyneris_v1/data.n5",
                                "BoundaryKey": "volumes/affinities/s1",
                                "MaxSegmentNumber": 32700,
                                "LabelSegmentation": False}}
    # cilia segmentation
    elif name == DYNAMIC_SEGMENTATIONS[1]:
        return {'PainteraProject': ['/g/kreshuk/data/arendt/platyneris_v1/data.n5',
                                    'volumes/paintera/proofread_cilia'],
                'TableUpdateFunction': 'make_cilia_tables'}
    # nuclei segmentation
    elif name == DYNAMIC_SEGMENTATIONS[2]:
        return {'PainteraProject': ['/g/kreshuk/data/arendt/platyneris_v1/data.n5',
                                    'volumes/paintera/nuclei'],
                'TableUpdateFunction': 'make_nuclei_tables'}
    else:
        return None


if __name__ == '__main__':
    with open(LUT_PATH, 'w') as f:
        json.dump(FILE_NAME_LUT, f, sort_keys=True, indent=2)
