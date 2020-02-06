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
ROOT = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data'

FILE_NAME_LUT = {}
IMAGE_PROPERTIES = {}


# we need to make the following update to names:
# - prospr -> gene names need to be replaced according to list
#             (including to two region names !) and be lowercase
# - prospr -> get rid of the '-MED' postfix
# - segmentations -> get rid of '-labels' postifx
def update_name_lut():
    global FILE_NAME_LUT

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
        properties = {}
        table_folder = 'tables/%s' % name

        # prospr: Color Magenta
        #         value range 0 - 1000
        if name.startswith('prospr'):
            if 'virtual-cells' in name:
                properties.update({'ColorMap': 'Glasbey', 'TableFolder': table_folder})
            else:
                properties.update({'Color': 'Magenta', 'MinValue': 0, 'MaxValue': 1000})

        # handle all special segmentations:
        # - dynamic and with tables:
        # -- cells
        # -- cilia
        # -- nuclei
        elif 'segmented-cells' in name:
            paintera_project = ''
            table_update_function = ''
            # TODO postprocessing options in Dynamic
            properties.update({'ColorMap': 'Glasbey',
                               'TableFolder': table_folder,
                               'Dynamic': {'PainteraProject': paintera_project,
                                           'TableUpdateFunction': table_update_function}})
        # - static but with tables:
        # -- chromatin
        # -- tissue
        # -- ganglia
        elif ('segmented-chromatin' in name
              or 'segmented-tissue' in name
              or 'segmented-ganglia' in name):
            properties.update({'ColorMap': 'Glasbey', 'TableFolder': table_folder})

        # TODO is white correct ?
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
    return FILE_NAME_LUT.get(file_name, None)


def get_image_properties(name):
    return IMAGE_PROPERTIES[name]


if __name__ == '__main__':
    x = json.dumps(FILE_NAME_LUT, sort_keys=True, indent=2)
    print(x)
    # with open('/home/pape/new_names.json', 'w') as f:
    #     json.dump(FILE_NAME_LUT, f, sort_keys=True, indent=2)
